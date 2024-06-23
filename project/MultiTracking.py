import argparse
from pathlib import Path

import cv2, pdb
import torch

from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT, WEIGHTS
from boxmot.utils import logger as LOGGER
from boxmot.utils.checks import TestRequirements
from boxmot.utils.torch_utils import select_device

__tr = TestRequirements()
__tr.check_packages(("ultralytics==8.0.124",))  # install

# from detectors import get_yolo_inferer
from ultralytics.yolo.data.utils import VID_FORMATS
from ultralytics.yolo.engine.model import TASK_MAP, YOLO
from ultralytics.yolo.utils import IterableSimpleNamespace, colorstr, ops
from ultralytics.yolo.utils.checks import check_imgsz

# from ultralytics.yolo.utils.files import increment_path
# from ultralytics.yolo.utils.plotting import save_one_box
# from utils import write_MOT_results
from PIL import Image, ImageDraw, ImageFont
from boxmot.utils import EXAMPLES
import os
from ultralytics.yolo.engine.results import Boxes, Results
import numpy as np
import torch
from ultralytics.yolo.utils import ops

os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"

from ultralytics.yolo.utils.ops import clip_boxes, xywh2xyxy, xyxy2xywh
import shutil


def save_one_box(
    xyxy, im, file=Path("im.jpg"), gain=1.02, pad=10, square=False, BGR=False, save=True
):
    """Save image crop as {file} with crop size multiple {gain} and {pad} pixels. Save and/or return crop."""
    b = xyxy2xywh(xyxy.view(-1, 4))  # boxes
    if square:
        b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # attempt rectangle to square
    b[:, 2:] = b[:, 2:] * gain + pad  # box wh * gain + pad
    xyxy = xywh2xyxy(b).long()
    clip_boxes(xyxy, im.shape)
    crop = im[
        int(xyxy[0, 1]) : int(xyxy[0, 3]),
        int(xyxy[0, 0]) : int(xyxy[0, 2]),
        :: (1 if BGR else -1),
    ]
    return crop


def write_MOT_results(txt_path, results, frame_idx, i):
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 1), -1)
    i = torch.full((nr_dets, 1), i)
    mot = torch.cat(
        [
            frame_idx,
            results.boxes.id.unsqueeze(1).to("cpu"),
            ops.xyxy2ltwh(results.boxes.xyxy).to("cpu"),
            results.boxes.conf.unsqueeze(1).to("cpu"),
            results.boxes.cls.unsqueeze(1).to("cpu"),
            dont_care,
        ],
        dim=1,
    )

    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create mot txt file
    txt_path.with_suffix(".txt").touch(exist_ok=True)

    with open(str(txt_path) + ".txt", "ab+") as f:  # append binary mode
        np.savetxt(
            f, mot.numpy(), fmt="%d"
        )  # save as ints instead of scientific notation


class YoloInterface:
    def inference(self, im):
        raise NotImplementedError("Subclasses must implement this method.")

    def postprocess(self, preds):
        raise NotImplementedError("Subclasses must implement this method.")

    def filter_results(self, i, predictor):
        if predictor.tracker_outputs[i].size != 0:
            # filter boxes masks and pose results by tracking results
            sorted_confs = predictor.tracker_outputs[i][:, 5].argsort()[::-1]
            predictor.tracker_outputs[i] = predictor.tracker_outputs[i][sorted_confs]
            yolo_confs = predictor.results[i].boxes.conf.cpu().numpy()
            tracker_confs = predictor.tracker_outputs[i][:, 5]
            mask = np.in1d(yolo_confs, tracker_confs)

            if predictor.results[i].masks is not None:
                predictor.results[i].masks = predictor.results[i].masks[mask]
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
            elif predictor.results[i].keypoints is not None:
                predictor.results[i].boxes = predictor.results[i].boxes[mask]
                predictor.results[i].keypoints = predictor.results[i].keypoints[mask]
        else:
            pass

    def overwrite_results(self, i, im0_shape, predictor):
        # overwrite bbox results with tracker predictions
        if predictor.tracker_outputs[i].size != 0:
            predictor.results[i].boxes = Boxes(
                # xyxy, (track_id), conf, cls
                boxes=torch.from_numpy(predictor.tracker_outputs[i]).to(
                    predictor.device
                ),
                orig_shape=im0_shape,  # (height, width)
            )

    def get_scaling_factors(self, im, im0s):
        # im to im0 factor for predictions
        im0_w = im0s[0].shape[1]
        im0_h = im0s[0].shape[0]
        im_w = im[0].shape[2]
        im_h = im[0].shape[1]
        w_r = im0_w / im_w
        h_r = im0_h / im_h

        return w_r, h_r

    def preds_to_yolov8_results(self, path, preds, im, im0s, predictor):
        predictor.results[0] = Results(
            path=path, boxes=preds, orig_img=im0s[0], names=predictor.model.names
        )
        return predictor.results


class Yolov8Strategy(YoloInterface):
    def __init__(self, model, device, args):
        self.model = model

    def inference(self, im):
        preds = self.model(im, augment=False, visualize=False)
        return preds

    def postprocess(self, path, preds, im, im0s, predictor):
        postprocessed_preds = predictor.postprocess(preds, im, im0s)
        return postprocessed_preds


def on_predict_start(predictor):
    predictor.trackers = []
    predictor.tracker_outputs = [None] * predictor.dataset.bs
    predictor.args.tracking_config = (
        ROOT / "boxmot" / "configs" / ("deepocsort" + ".yaml")
    )
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.args.tracking_method,
            predictor.args.tracking_config,
            predictor.args.reid_model,
            predictor.device,
            predictor.args.half,
            predictor.args.per_class,
        )
        predictor.trackers.append(tracker)


def inital_args(checkpoint_dir, video_path, save_dir, device="0", step=1):
    from pathlib import PosixPath
    save_name = os.path.splitext(video_path.split("/")[-1])[0]
    opt={
    'yolo_model': PosixPath(checkpoint_dir + "/" + "yolov8n-seg.pt"),
    'reid_model': PosixPath(checkpoint_dir + "/" + "mobilenetv2_x1_4_dukemtmcreid.pt"),
    'tracking_method': 'deepocsort',
    'source': video_path,
    'imgsz': [640],
    'conf': 0.5,
    'iou': 0.7,
    'device': device,
    'show': False,
    'save': True,
    'classes': None,
    'project': save_dir,
    'name': save_name,
    'exist_ok': False,
    'half': False,
    'vid_stride': 30,
    'show_labels': False,
    'show_conf': False,
    'save_txt': True,
    'save_id_crops': True,
    'save_mot': True,
    'line_width': None,
    'per_class': False
    }
    return opt


    # save_name = os.path.splitext(video_path.split("/")[-1])[0]

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--yolo-model",
    #     type=Path,
    #     default=checkpoint_dir + "/" + "yolov8n-seg.pt",
    #     help="model.pt path(s)",
    # )
    # parser.add_argument(
    #     "--reid-model",
    #     type=Path,
    #     default=checkpoint_dir + "/" + "mobilenetv2_x1_4_dukemtmcreid.pt",
    # )
    # parser.add_argument(
    #     "--tracking-method",
    #     type=str,
    #     default="deepocsort",
    #     help="deepocsort, botsort, strongsort, ocsort, bytetrack",
    # )
    # parser.add_argument(
    #     "--source", type=str, default=video_path, help="file/dir/URL/glob, 0 for webcam"
    # )
    # parser.add_argument(
    #     "--imgsz",
    #     "--img",
    #     "--img-size",
    #     nargs="+",
    #     type=int,
    #     default=[640],
    #     help="inference size h,w",
    # )
    # parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    # parser.add_argument(
    #     "--iou",
    #     type=float,
    #     default=0.7,
    #     help="intersection over union (IoU) threshold for NMS",
    # )
    # parser.add_argument(
    #     "--device", default=device, help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    # )
    # parser.add_argument(
    #     "--show",
    #     action="store_true",
    #     default=False,
    #     help="display tracking video results",
    # )
    # parser.add_argument(
    #     "--save", action="store_true", default=True, help="save video tracking results"
    # )
    # # # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    # parser.add_argument(
    #     "--classes",
    #     nargs="+",
    #     type=int,
    #     help="filter by class: --classes 0, or --classes 0 2 3",
    # )
    # parser.add_argument(
    #     "--project", default=save_dir, help="save results to project/name"
    # )
    # parser.add_argument(
    #     "--name", default=save_name, help="save results to project/name"
    # )
    # parser.add_argument(
    #     "--exist-ok",
    #     action="store_true",
    #     help="existing project/name ok, do not increment",
    # )
    # parser.add_argument(
    #     "--half", action="store_true", help="use FP16 half-precision inference"
    # )
    # parser.add_argument(
    #     "--vid-stride", type=int, default=step, help="video frame-rate stride"
    # )
    # parser.add_argument(
    #     "--show-labels",
    #     action="store_false",
    #     default=False,
    #     help="hide labels when show",
    # )
    # parser.add_argument(
    #     "--show-conf",
    #     action="store_false",
    #     default=False,
    #     help="hide confidences when show",
    # )
    # parser.add_argument(
    #     "--save-txt",
    #     action="store_true",
    #     default=True,
    #     help="save tracking results in a txt file",
    # )
    # parser.add_argument(
    #     "--save-id-crops",
    #     action="store_true",
    #     default=True,
    #     help="save each crop to its respective id folder",
    # )
    # parser.add_argument(
    #     "--save-mot",
    #     action="store_true",
    #     default=True,
    #     help="save tracking results in a single txt file",
    # )
    # parser.add_argument(
    #     "--line-width",
    #     default=None,
    #     type=int,
    #     help="The line width of the bounding boxes. If None, it is scaled to the image size.",
    # )
    # parser.add_argument(
    #     "--per-class", action="store_true", help="not mix up classes when tracking"
    # )
    # opt = parser.parse_args()
    # return vars(opt)


@torch.no_grad()
def detect_by_path(checkpoint_dir, video_path, save_dir, device="0", step=1):
    track_args = inital_args(checkpoint_dir, video_path, save_dir, device, step)
    model = YOLO(track_args["yolo_model"] if "v8" in str(track_args["yolo_model"]) else "yolov8n")
    overrides = model.overrides.copy()
    model.predictor = TASK_MAP[model.task][3](
        overrides=overrides, _callbacks=model.callbacks
    )

    # extract task predictor
    predictor = model.predictor

    # combine default predictor args with custom, preferring custom
    combined_args = {**predictor.args.__dict__, **track_args}
    # overwrite default args
    predictor.args = IterableSimpleNamespace(**combined_args)
    predictor.args.device = select_device(track_args["device"])
    LOGGER.info(track_args)

    # setup source and model
    if not predictor.model:
        predictor.setup_model(model=model.model, verbose=True)
    predictor.setup_source(predictor.args.source)

    predictor.args.imgsz = check_imgsz(
        predictor.args.imgsz, stride=model.model.stride, min_dim=2
    )  # check image size
    # predictor.save_dir = increment_path(Path(predictor.args.project) /
    #                                     predictor.args.name, exist_ok=predictor.args.exist_ok)

    predictor.save_dir = Path(predictor.args.project) / predictor.args.name
    if os.path.exists(predictor.save_dir):
        shutil.rmtree(predictor.save_dir)

    os.makedirs(predictor.save_dir)

    # Check if save_dir/ label file exists
    if predictor.args.save or predictor.args.save_txt:
        (
            predictor.save_dir / "labels"
            if predictor.args.save_txt
            else predictor.save_dir
        ).mkdir(parents=True, exist_ok=True)

    # Warmup model
    if not predictor.done_warmup:
        predictor.model.warmup(
            imgsz=(
                1
                if predictor.model.pt or predictor.model.triton
                else predictor.dataset.bs,
                3,
                *predictor.imgsz,
            )
        )
        predictor.done_warmup = True
    predictor.seen, predictor.windows, predictor.batch, predictor.profilers = (
        0,
        [],
        None,
        (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile()),
    )

    predictor.add_callback("on_predict_start", on_predict_start)
    predictor.run_callbacks("on_predict_start")
    # get yolo class based on model name
    Yolo = Yolov8Strategy

    # initialize class
    model = Yolo(
        model=model.predictor.model
        if "v8" in str(track_args["yolo_model"])
        else track_args["yolo_model"],
        device=predictor.device,
        args=predictor.args,
    )

    for frame_idx, batch in enumerate(predictor.dataset):
        predictor.run_callbacks("on_predict_batch_start")
        predictor.batch = batch
        path, im0s, vid_cap, s = batch

        n = len(im0s)
        predictor.results = [None] * n

        # Preprocess
        with predictor.profilers[0]:
            im = predictor.preprocess(im0s)

        # Inference
        with predictor.profilers[1]:
            preds = model.inference(im=im)

        # Postprocess moved to MultiYolo
        with predictor.profilers[2]:
            predictor.results = model.postprocess(path, preds, im, im0s, predictor)
        predictor.run_callbacks("on_predict_postprocess_end")

        # Visualize, save, write results
        n = len(im0s)
        for i in range(n):
            if (
                predictor.dataset.source_type.tensor
            ):  # skip write, show and plot operations if input is raw tensor
                continue
            p, im0 = path[i], im0s[i].copy()
            p = Path(p)

            with predictor.profilers[3]:
                # get raw bboxes tensor
                dets = predictor.results[i].boxes.data
                # get tracker predictions
                predictor.tracker_outputs[i] = predictor.trackers[i].update(
                    dets.cpu().detach().numpy(), im0
                )
            predictor.results[i].speed = {
                "preprocess": predictor.profilers[0].dt * 1e3 / n,
                "inference": predictor.profilers[1].dt * 1e3 / n,
                "postprocess": predictor.profilers[2].dt * 1e3 / n,
                "tracking": predictor.profilers[3].dt * 1e3 / n,
            }

            # filter boxes masks and pose results by tracking results
            model.filter_results(i, predictor)
            # overwrite bbox results with tracker predictions
            model.overwrite_results(i, im0.shape[:2], predictor)

            # write inference results to a file or directory
            if (
                predictor.args.verbose
                or predictor.args.save
                or predictor.args.save_txt
                or predictor.args.show
                or predictor.args.save_id_crops
            ):
                s += predictor.write_results(i, predictor.results, (p, im, im0))
                predictor.txt_path = Path(predictor.txt_path)

                # write MOT specific results
                if predictor.args.source.endswith(VID_FORMATS):
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                # mot txt called the same as the parent name to perform inference on
                elif "MOT16" or "MOT17" or "MOT20" in predictor.args.source:
                    predictor.MOT_txt_path = (
                        predictor.txt_path.parent / p.parent.parent.name
                    )
                # mot txt called the same as the parent name to perform inference on
                else:
                    predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name

                if predictor.tracker_outputs[i].size != 0 and predictor.args.save_mot:
                    write_MOT_results(
                        predictor.MOT_txt_path,
                        predictor.results[i],
                        frame_idx,
                        i,
                    )

                # if predictor.args.save_id_crops:
                #     for d in predictor.results[i].boxes:

                #         crop = save_one_box(
                #             d.xyxy,
                #             im0.copy(),
                #             file= None,
                #             BGR=True
                #         )

                #         if d.id is not None and d.cls is not None :
                #             tmp_file = predictor.save_dir / 'crops' /str(int(d.cls.cpu().numpy().item())) /str(int(d.id.cpu().numpy().item())) / f'{p.stem}.jpg'
                #             tmp_file.parent.mkdir(parents=True, exist_ok=True)  # make directory
                #             new_path = tmp_file.with_name(tmp_file.stem+'_'+str(frame_idx+1)+ tmp_file.suffix)
                #             Image.fromarray(crop[..., ::-1]).save(new_path, quality=95, subsampling=0)  # save RGB

                if predictor.args.save_id_crops:
                    for id_i in range(0, len(predictor.results[i].boxes)):
                        d = predictor.results[i].boxes[id_i]
                        crop = save_one_box(d.xyxy, im0.copy(), file=None, BGR=True)

                        if d.id is not None and d.cls is not None:
                            tmp_file = (
                                predictor.save_dir
                                / "crops"
                                / str(int(d.cls.cpu().numpy().item()))
                                / str(int(d.id.cpu().numpy().item()))
                                / f"{p.stem}.jpg"
                            )
                            tmp_file.parent.mkdir(
                                parents=True, exist_ok=True
                            )  # make directory
                            new_path = tmp_file.with_name(
                                tmp_file.stem
                                + "_"
                                + str(frame_idx + 1)
                                + tmp_file.suffix
                            )
                            Image.fromarray(crop[..., ::-1]).save(
                                new_path, quality=95, subsampling=0
                            )  # save RGB

                            seg_tmp_file = (
                                predictor.save_dir
                                / "segs"
                                / str(int(d.cls.cpu().numpy().item()))
                                / str(int(d.id.cpu().numpy().item()))
                                / f"{p.stem}.jpg"
                            )
                            seg_tmp_file.parent.mkdir(
                                parents=True, exist_ok=True
                            )  # make directory
                            seg_new_path = seg_tmp_file.with_name(
                                seg_tmp_file.stem
                                + "_"
                                + str(frame_idx + 1)
                                + seg_tmp_file.suffix
                            )
                            id_shape = (
                                predictor.results[i]
                                .masks.data[id_i]
                                .squeeze(0)
                                .cpu()
                                .numpy()
                                .shape
                            )
                            if predictor.results[i].masks.data[id_i] is not None:
                                predictor.results[i].orig_shape
                                res = cv2.resize(
                                    predictor.results[i]
                                    .masks.data[id_i]
                                    .squeeze(0)
                                    .cpu()
                                    .numpy()
                                    * 255,
                                    (id_shape[1], id_shape[0]),
                                )

                                cv2.imwrite(str(seg_new_path), res)
                            else:
                                cv2.imwrite(str(seg_new_path), np.zeros(id_shape))

            # display an image in a window using OpenCV imshow()
            if predictor.args.show and predictor.plotted_img is not None:
                predictor.show(p.parent)

            # save video predictions
            if predictor.args.save and predictor.plotted_img is not None:
                predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

        predictor.run_callbacks("on_predict_batch_end")

        # print time (inference-only)
        if predictor.args.verbose:
            s_t = f"YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms"
            LOGGER.info(f"{s}{s_t}")

    # Release assets
    if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
        predictor.vid_writer[-1].release()  # release final video writer

    # Print results
    if predictor.args.verbose and predictor.seen:
        t = tuple(
            x.t / predictor.seen * 1e3 for x in predictor.profilers
        )  # speeds per image
        LOGGER.info(
            f"Speed: %.1fms preproc, %.1fms inference, %.1fms postproc, %.1fms tracking per image at shape "
            f"{(1, 3, *predictor.args.imgsz)}" % t
        )
    if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
        nl = len(list(predictor.save_dir.glob("labels/*.txt")))  # number of labels
        s = (
            f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}"
            if predictor.args.save_txt
            else ""
        )
        LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

    predictor.run_callbacks("on_predict_end")

    return predictor.save_dir


# if __name__ == "__main__":
#     checkpoint_dir = "/data02/lxd/py_project/videogpt/videogpt/checkpoints"
#     video_path = "/data02/lxd/py_project/videogpt/videogpt/demo/test/demo1.mp4"
#     save_dir = "/data02/lxd/py_project/videogpt/videogpt/demo/track_res"
#     detect_by_path(checkpoint_dir, video_path, save_dir,device='1',step=10)
