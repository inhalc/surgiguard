# SurgiGuard design

## Core problem

Frame-wise segmenters can produce temporally unstable masks under motion, occlusion, irrigation,
or short acquisition artifacts. Blind smoothing reduces flicker but can delay a real change. The
public implementation therefore treats stability and responsiveness as an explicit control tradeoff.

## Technical choices

1. **Keep the segmenter fixed.** Any authorized model can supply the current probability map. This
   isolates temporal reliability from model training, at the cost of not correcting systematic
   semantic errors in the base model.
2. **Align and qualify history.** Farnebäck optical flow maps stored probabilities into the target
   frame. Forward–backward consistency and bounds produce confidence weights. This is inexpensive
   and inspectable, but large occlusions and weak texture can still invalidate history.
3. **Gate history per pixel.** A robust history reference anchors stable regions while current-frame
   disagreement releases the update. This responds faster than uniform smoothing, but its operating
   point must be selected for the acquisition setting.

## Implementation boundary

The controller stores source frames, probability maps, and monotonically increasing timestamps.
The API isolates state by `stream_id`, serializes updates within a stream, bounds the number of
active streams, and supports reset. Monitoring reports observable mask dynamics—including area
growth—not a clinical outcome or directional risk prediction. The bundled demo and screenshots are
synthetic and deterministic.
