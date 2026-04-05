import cv2
import numpy as np

def detect_features(gray_frame, max_corners=150, quality=0.01,
                    min_distance=10, block_size=7):
    return cv2.goodFeaturesToTrack(
        gray_frame,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=block_size,
    )


def _build_pyramid(img, levels):
    pyr = [img]
    for _ in range(levels):
        pyr.append(cv2.pyrDown(pyr[-1]))
    return pyr


def _lk_single_level(prev, curr, pts, win_half=10, tau=1e-2):
    Ix = cv2.Sobel(prev, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(prev, cv2.CV_64F, 0, 1, ksize=3)
    It = curr.astype(np.float64) - prev.astype(np.float64)

    h, w = prev.shape
    new_pts = pts.copy().astype(np.float64)
    valid   = np.zeros(len(pts), dtype=bool)

    for i, (x, y) in enumerate(pts):
        x, y = int(round(x)), int(round(y))

        x0, x1 = max(0, x - win_half), min(w, x + win_half + 1)
        y0, y1 = max(0, y - win_half), min(h, y + win_half + 1)

        ix = Ix[y0:y1, x0:x1].ravel()
        iy = Iy[y0:y1, x0:x1].ravel()
        it = It[y0:y1, x0:x1].ravel()

        ATA = np.array([[np.dot(ix,ix), np.dot(ix,iy)],
                        [np.dot(ix,iy), np.dot(iy,iy)]])

        if np.linalg.eigvalsh(ATA)[0] < tau:
            continue

        ATb = np.array([-np.dot(ix, it), -np.dot(iy, it)])
        flow = np.linalg.solve(ATA, ATb)

        new_pts[i] = pts[i] + flow
        valid[i]   = True

    return new_pts.astype(np.float32), valid


def lucas_kanade_flow(prev_gray, curr_gray, prev_pts,
                      win_size=(21, 21), max_level=3,
                      eps=0.03, max_iter=30):
    win_half = win_size[0] // 2
    prev_pyr = _build_pyramid(prev_gray, max_level)
    curr_pyr = _build_pyramid(curr_gray, max_level)

    pts  = prev_pts.reshape(-1, 2).astype(np.float32)
    flow = np.zeros_like(pts)

    for level in range(max_level, -1, -1):
        s       = 1.0 / (2 ** level)
        pts_l   = pts * s
        guess_l = pts_l + flow

        new_pts, valid = _lk_single_level(
            prev_pyr[level], curr_pyr[level],
            guess_l, win_half=win_half
        )

        delta = new_pts - pts_l
        flow  = delta * 2 if level > 0 else delta

    curr_pts  = pts + flow
    good_prev = pts[valid]
    good_curr = curr_pts[valid]
    flow_vecs = good_curr - good_prev
    return good_prev, good_curr, flow_vecs


def draw_sparse_flow(frame, good_prev, good_curr,
                     trail_canvas, point_color=(0, 100, 255),
                     arrow_color=(255, 50, 200), radius=4,
                     scale=2.0):
    out = frame.copy()

    for pt0, pt1 in zip(good_prev, good_curr):
        x0, y0 = pt0.astype(int)
        x1, y1 = pt1.astype(int)

        cv2.line(trail_canvas, (x0, y0), (x1, y1), point_color, 1)

        dx, dy = (pt1 - pt0) * scale
        x2, y2 = int(x0 + dx), int(y0 + dy)
        cv2.arrowedLine(out, (x0, y0), (x2, y2),
                        arrow_color, 1, tipLength=0.4)
        cv2.circle(out, (x1, y1), radius, point_color, -1)

    out = cv2.add(out, trail_canvas)
    return out


def dense_optical_flow(prev_gray, curr_gray):
    small_prev = cv2.resize(prev_gray, (0, 0), fx=0.5, fy=0.5)
    small_curr = cv2.resize(curr_gray, (0, 0), fx=0.5, fy=0.5)

    flow = cv2.calcOpticalFlowFarneback(
        small_prev, small_curr, None,
        pyr_scale=0.5, levels=2, winsize=11,
        iterations=2, poly_n=5, poly_sigma=1.1, flags=0,
    )

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv = np.zeros((*small_prev.shape, 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    bgr_small = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.resize(bgr_small, (prev_gray.shape[1], prev_gray.shape[0]))


def run_on_video(video_path, win_size=(25, 25), max_level=2,
                 redetect_interval=15, scale=0.35):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Cannot open {video_path}"

    wait_ms = 1

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read first frame.")

    frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

    prev_gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_pts     = detect_features(prev_gray)
    trail_canvas = np.zeros_like(frame)
    t = 0

    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
        t += 1

        curr_frame = cv2.resize(curr_frame, (0, 0), fx=scale, fy=scale)
        curr_gray  = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        if prev_pts is not None and len(prev_pts) > 0:
            good_prev, good_curr, _ = lucas_kanade_flow(
                prev_gray, curr_gray, prev_pts,
                win_size=win_size, max_level=max_level
            )
        else:
            good_prev = good_curr = np.empty((0, 2))

        sparse_vis = draw_sparse_flow(curr_frame, good_prev,
                                      good_curr, trail_canvas)
        dense_vis  = dense_optical_flow(prev_gray, curr_gray)

        cv2.putText(sparse_vis, f"Sparse LK (frame {t})",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(dense_vis, f"Dense Farneback (frame {t})",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        h, w = sparse_vis.shape[:2]
        dense_vis = cv2.resize(dense_vis, (w, h))
        combined = np.vstack([sparse_vis, dense_vis])
        cv2.imshow("Optical Flow  |  press Q to quit", combined)

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

        if t % redetect_interval == 0 or len(good_curr) < 20:
            trail_canvas[:] = 0
            prev_pts = detect_features(curr_gray)
        else:
            prev_pts = good_curr.reshape(-1, 1, 2).astype(np.float32)

        prev_gray = curr_gray.copy()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:  python t1.py <path_to_your_video.mp4>")
        print("Example: python t1.py OPTICAL_FLOW.mp4")
        sys.exit(1)

    run_on_video(sys.argv[1])
