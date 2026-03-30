/*
 * Task 3: Noisy LiDAR-Based Dynamic Obstacle Avoidance
 * Noise Handling : Spatial Median Filter  (window = 5, tuned for 50 rays)
 * Navigation     : Grid-sampled waypoints through free space, visited in
 *                  nearest-neighbour order. VFH+ scoring is directed toward
 *                  the active waypoint instead of just "forward".
 *
 * Prerequisites (simulation.hpp):
 *   1. genobs(5, 0.1, 10)   in constructor
 *   2. raycastagent(players[i], 0.05f)   in update()
 *   3. rays = 50   (already done)
 */

#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"

// ── Tunable parameters ───────────────────────────────────────────────────── //
namespace cfg {
    constexpr int   STATIC_COUNT    = 3;

    // VFH+ histogram
    constexpr ftype HIST_DECAY      = 3.0f;   // stronger obstacle repulsion
    constexpr ftype WP_ALPHA        = 0.80f;  // waypoint-direction attraction
    constexpr ftype STEER_BETA      = 0.15f;  // small heading-change penalty

    // Clearance thresholds
    constexpr ftype SAFE_DIST       = 0.35f;
    constexpr ftype BRAKE_DIST      = 0.15f;
    constexpr ftype EMERGENCY_DIST  = 0.08f;  // hard-brake below this
    constexpr ftype ACCEL_SCALE     = 3.5f;
    constexpr ftype SPEED_SMOOTH    = 0.12f;

    // PD steering  (tuned for 50-ray resolution)
    constexpr ftype STEER_KP        = 0.55f;
    constexpr ftype STEER_KD        = 0.15f;

    // Median filter
    constexpr int   MED_HALF_W      = 2;      // window = 5 (suits 50 rays)

    // Waypoint system
    constexpr ftype WP_GRID_STEP    = 0.22f;  // grid spacing in [-0.85, 0.85]
    constexpr ftype WP_ARRIVE_DIST  = 0.06f;  // radius to count wp as reached
    constexpr ftype WP_OBS_MARGIN   = 0.04f;  // extra clearance when filtering
}
// ─────────────────────────────────────────────────────────────────────────── //

// ── Median filter ────────────────────────────────────────────────────────── //
inline ftype windowMedian(const array<ftype, rays>& d, int centre) {
    constexpr int W = 2 * cfg::MED_HALF_W + 1;
    ftype buf[W];
    for (int k = 0; k < W; ++k)
        buf[k] = d[(centre - cfg::MED_HALF_W + k + rays) % rays];
    // insertion sort — W is tiny
    for (int i = 1; i < W; ++i) {
        ftype key = buf[i]; int j = i - 1;
        while (j >= 0 && buf[j] > key) { buf[j+1] = buf[j]; --j; }
        buf[j+1] = key;
    }
    return buf[W / 2];
}

// ── Angle normalisation to (-PI, PI] ─────────────────────────────────────── //
inline ftype normAngle(ftype a) {
    while (a >  PI) a -= 2.0f * PI;
    while (a < -PI) a += 2.0f * PI;
    return a;
}

// ── Nearest-neighbour waypoint ordering from a seed point ────────────────── //
static vector<point> nnOrder(vector<point> pts, point seed) {
    vector<point> out;
    out.reserve(pts.size());
    while (!pts.empty()) {
        int    best = 0;
        ftype  bd   = dist(seed, pts[0]);
        for (int i = 1; i < (int)pts.size(); ++i) {
            ftype d = dist(seed, pts[i]);
            if (d < bd) { bd = d; best = i; }
        }
        seed = pts[best];
        out.push_back(pts[best]);
        pts.erase(pts.begin() + best);
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────── //

int main() {

    // ── Persistent controller state ───────────────────────────────────── //
    static ftype prev_steer_err = 0.0f;
    static ftype filtered_accel = 0.0f;

    // ── Waypoint state ────────────────────────────────────────────────── //
    static vector<point> waypoints;
    static int           wp_idx      = 0;
    static bool          wp_ready    = false;

    // ── Agent ─────────────────────────────────────────────────────────── //
    agent myagent;

    myagent.calculate_1 =
        [](const envmap& curmap,
           const array<pair<point, point>, playercount>& playerdata,
           const array<point, rays>& raycasts,
           const agent& curplayer,
           ftype& a, ftype& steer)
    {
        const point& mypos = playerdata[0].first;

        // ── One-time waypoint generation ─────────────────────────────── //
        if (!wp_ready) {
            // Inflate each obstacle polygon for the containment test
            // (we use cfg::WP_OBS_MARGIN simply by testing a slightly
            //  enlarged set of candidate points instead of inflating shapes)
            vector<point> cands;
            for (ftype x = -0.85f; x <= 0.851f; x += cfg::WP_GRID_STEP)
                for (ftype y = -0.85f; y <= 0.851f; y += cfg::WP_GRID_STEP) {
                    point p(x, y);
                    bool free = true;
                    // Check p and a small ring of nearby points for margin
                    const ftype M = cfg::WP_OBS_MARGIN;
                    array<point,5> probes = {p,
                        point(x+M, y), point(x-M, y),
                        point(x, y+M), point(x, y-M)};
                    for (auto& pr : probes) {
                        for (auto& obs : curmap)
                            if (contains(obs, pr)) { free = false; break; }
                        if (!free) break;
                    }
                    if (free) cands.push_back(p);
                }
            waypoints = nnOrder(cands, mypos);
            wp_idx    = 0;
            wp_ready  = true;
        }

        // ── Advance to next waypoint when close enough ────────────────── //
        if (!waypoints.empty()) {
            while (wp_idx < (int)waypoints.size() &&
                   dist(mypos, waypoints[wp_idx]) < cfg::WP_ARRIVE_DIST)
                ++wp_idx;
            if (wp_idx >= (int)waypoints.size())
                wp_idx = (int)waypoints.size() - 1;  // hold last wp
        }
// 1. Calculate current heading legally from the direction vector
        ftype current_heading = atan2(playerdata[0].second.y, playerdata[0].second.x);

        // 2. Direction from current position to active waypoint (world frame)
        ftype wp_world_angle = (waypoints.empty())
            ? current_heading
            : atan2(waypoints[wp_idx].y - mypos.y,
                    waypoints[wp_idx].x - mypos.x);

        // 3. Angle to waypoint relative to current heading
        ftype rel_wp = normAngle(wp_world_angle - current_heading);

        // ── Step 1. Raw LiDAR distances ───────────────────────────────── //
        array<ftype, rays> raw_dist{};
        for (int i = 0; i < rays; i++)
            raw_dist[i] = dist(mypos, raycasts[i]);

        // ── Step 2. Median filter (window=5, wrap-around) ─────────────── //
        array<ftype, rays> clean_dist{};
        for (int i = 0; i < rays; i++)
            clean_dist[i] = windowMedian(raw_dist, i);

        // ── Step 3. Polar obstacle-cost histogram ─────────────────────── //
        array<ftype, rays> hist{};
        for (int i = 0; i < rays; i++) {
            ftype d = max(clean_dist[i], 1e-3f);
            hist[i] = pow(1.0f / d, cfg::HIST_DECAY);
        }

        // ── Step 4. Gaussian smoothing (3-tap, wrap-around) ───────────── //
        array<ftype, rays> smooth{};
        for (int i = 0; i < rays; i++) {
            smooth[i] = 0.25f * hist[(i-1+rays)%rays]
                      + 0.50f * hist[i]
                      + 0.25f * hist[(i+1)%rays];
        }

        // ── Step 5. VFH+ scoring toward active waypoint ───────────────── //
        //    Each ray is at heading-relative angle: 2π·i/rays
        //    Score = -obstacle_cost
        //           + WP_ALPHA  * cos(ray_angle - wp_relative_angle)
        //           - STEER_BETA * |ray_angle|   (prefer small course changes)
        int   best_ray   = 0;
        ftype best_score = -1e9f;

        for (int i = 0; i < rays; i++) {
            ftype ray_ang = 2.0f * PI * i / (ftype)rays;
            if (ray_ang > PI) ray_ang -= 2.0f * PI;  // normalise to (-π,π]

            ftype delta_to_wp = normAngle(ray_ang - rel_wp);

            ftype score = -smooth[i]
                        + cfg::WP_ALPHA  * cos(delta_to_wp)
                        - cfg::STEER_BETA * abs(ray_ang);

            if (score > best_score) { best_score = score; best_ray = i; }
        }

        // ── Step 6. PD steering toward best ray ───────────────────────── //
        ftype steer_err = 2.0f * PI * best_ray / (ftype)rays;
        if (steer_err > PI) steer_err -= 2.0f * PI;

        ftype d_err = steer_err - prev_steer_err;
        prev_steer_err = steer_err;

        steer = cfg::STEER_KP * steer_err + cfg::STEER_KD * d_err;

        // ── Step 7. Speed — forward ±60° cone + emergency brake ──────── //
        const int cone = rays / 6;   // ≈ 8 rays for ±57.6° with 50 rays
        ftype cone_dist = 0.0f;
        ftype min_front = 1e9f;

        for (int i = -cone; i <= cone; i++) {
            ftype cd = clean_dist[(i + rays) % rays];
            cone_dist += cd;
            if (cd < min_front) min_front = cd;
        }
        cone_dist /= (ftype)(2 * cone + 1);

        // Hard emergency brake — override everything if too close
        if (min_front < cfg::EMERGENCY_DIST) {
            filtered_accel = -acceldelta * cfg::ACCEL_SCALE * 2.0f;
            a = filtered_accel;
            return;
        }

        ftype target_a;
        if (cone_dist > cfg::SAFE_DIST) {
            target_a =  acceldelta * cfg::ACCEL_SCALE;
        } else if (cone_dist < cfg::BRAKE_DIST) {
            target_a = -acceldelta * cfg::ACCEL_SCALE;
        } else {
            ftype t = (cone_dist - cfg::BRAKE_DIST)
                    / (cfg::SAFE_DIST - cfg::BRAKE_DIST);
            target_a = acceldelta * cfg::ACCEL_SCALE * (2.0f * t - 1.0f);
        }

        filtered_accel = cfg::SPEED_SMOOTH * target_a
                       + (1.0f - cfg::SPEED_SMOOTH) * filtered_accel;
        a = filtered_accel;
    };

    // ── Simulation setup ──────────────────────────────────────────────── //
    array<agent, playercount> myagents;
    for (int i = 0; i < playercount; i++) myagents[i] = myagent;

    simulationinstance s(myagents, 60.0f);

    // ── Obstacle movement specifiers ───────────────────────────────────── //
    const int num_obs = (int)s.mp.size() - 1;
    vector<vector<point>> initial_obs(s.mp.begin(), s.mp.begin() + num_obs);

    for (int i = 0; i < num_obs; i++) {
        if (i < cfg::STATIC_COUNT) {
            s.movementspecifier[i] = [](vector<point>&, const ftype) {};
        } else {
            auto  init   = initial_obs[i];
            ftype phase  = 2.0f * PI * i / (ftype)(num_obs - cfg::STATIC_COUNT);
            ftype amp    = 0.04f + 0.015f * (i % 3);
            ftype period = 2.50f + 0.400f * (i % 5);

            s.movementspecifier[i] =
                [init, phase, amp, period](vector<point>& obs, const ftype t) {
                    ftype dx = amp * sin(2.0f * PI * t / period          + phase);
                    ftype dy = amp * cos(2.0f * PI * t / (period * 1.3f) + phase);
                    obs = point(dx, dy) + init;
                };
        }
    }

    s.humanmode  = false;
    s.visualmode = true;
    s.run();

    return 0;
}