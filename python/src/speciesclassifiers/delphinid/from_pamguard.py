"""Detection frames straight from PAMGuard binaries and an annotations CSV."""

from ..io import label_detections, periods, read_annotations, read_clicks, read_whistles
from .frames import AcousticParams, click_frames, whistle_frames


def make_frames(binary_folder, annotations, params: AcousticParams, sample_rate=None, progress=True):
    """Read the detections in each annotated period and turn them into detection frames.

    ``annotations``: CSV path or DataFrame (see ``speciesclassifiers.io.annotations``).
    ``sample_rate``: leave as None to work it out from the binaries.
    Returns ``(frames, info)``; ``info`` has detection counts and the recording settings used.
    """
    ann = read_annotations(annotations)
    if params.voc_type == "click":
        det, sr = read_clicks(binary_folder, periods(ann), sample_rate, progress=progress)
        settings = {"sample_rate": sr}
    else:
        det, settings = read_whistles(binary_folder, periods(ann), sample_rate, progress=progress)
        sr = settings["sample_rate"]
    labelled = label_detections(det, ann)
    frames = click_frames(labelled, params, sr) if params.voc_type == "click" else whistle_frames(labelled, params)
    info = {
        **settings,
        "n_detections_read": len(det),
        "n_detections_annotated": len(labelled),
        "n_frames": len(frames),
        "frames_per_label": frames["label"].value_counts().to_dict() if len(frames) else {},
        "events_without_frames": sorted(set(ann["event_id"]) - set(frames.get("event_id", []))),
    }
    if progress:
        print(f"{info['n_detections_annotated']} of {info['n_detections_read']} detections fall in annotated "
              f"periods -> {info['n_frames']} frames. Settings: {settings}")
        if info["events_without_frames"]:
            print(f"No frames for events: {info['events_without_frames']}")
    return frames, info
