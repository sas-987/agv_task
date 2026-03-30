/*
 * Task 2: Dynamic Obstacle Avoidance Controller  —  VFH+ planner
 *
 * Environment
 * -----------
 *   10 obstacles (boundary wall always static):
 *     Indices [0 .. STATIC_COUNT-1]  → fixed in place (parked / infrastructure)
 *     Indices [STATIC_COUNT .. N-2]  → Lissajous sinusoidal motion
 *
 * Controller — Vector Field Histogram+ (VFH+)
 * -------------------------------------------
 *   1. Build a polar obstacle-cost histogram from raycasts (inverse-distance).
 *   2. Smooth histogram with a 3-sector Gaussian kernel to avoid boundary flicker.
 *   3. Score each direction:  -cost  +  alpha*cos(delta)  -  beta*|delta|
 *   4. PD steer toward best direction (derivative term damps oscillation).
 *   5. Speed: average clearance in front +-60 cone,
 *      exponential low-pass filter on acceleration to reduce jerk.
 */

#include "draw.hpp"
#include "geometry.hpp"
#include "simulation.hpp"

// ── Tunable parameters ──────────────────────────────────────────────────── //
namespace cfg {
    constexpr int   STATIC_COUNT  = 4;     // obstacles kept fixed

    constexpr ftype HIST_DECAY    = 2.5f;  // inverse-dist cost exponent
    constexpr ftype KERNEL[3]     = {0.25f, 0.50f, 0.25f};  // 3-tap Gaussian

    constexpr ftype FORWARD_ALPHA = 0.50f; // reward facing forward
    constexpr ftype STEER_BETA    = 0.20f; // penalise large heading changes

    constexpr ftype SAFE_DIST     = 0.30f;
    constexpr ftype BRAKE_DIST    = 0.12f;
    constexpr ftype ACCEL_SCALE   = 3.0f;
    constexpr ftype SPEED_SMOOTH  = 0.15f; // low-pass alpha

    constexpr ftype STEER_KP      = 0.40f;
    constexpr ftype STEER_KD      = 0.10f;
}
// ──────────────────────────────────────────────────────────────────────────── //

int main() {

    // ------------------------------------------------------------------ //
    //  Persistent planner state — plain statics, no heap allocation needed
    // ------------------------------------------------------------------ //
    static ftype prev_steer_err  = 0.0f;
    static ftype filtered_accel  = 0.0f;

    // ------------------------------------------------------------------ //
    //  VFH+ Controller
    // ------------------------------------------------------------------ //
    agent myagent;

    myagent.calculate_1 =
        [](const envmap& curmap,
           const array<pair<point, point>, playercount>& playerdata,
           const array<point, rays>& raycasts,
           const agent& curplayer,
           ftype& a, ftype& steer) {

            const point& mypos = playerdata[0].first;

            // ── 1. Raw polar obstacle-cost histogram ─────────────────── //
            array<ftype, rays> hist{};
            for (int i = 0; i < rays; i++) {
                ftype d = dist(mypos, raycasts[i]);
                hist[i] = (d > 1e-3f) ? pow(1.0f / d, cfg::HIST_DECAY) : 1e6f;
            }

            // ── 2. Smooth histogram with wrap-around ─────────────────── //
            array<ftype, rays> smooth{};
            for (int i = 0; i < rays; i++) {
                smooth[i] = cfg::KERNEL[0] * hist[(i - 1 + rays) % rays]
                          + cfg::KERNEL[1] * hist[i]
                          + cfg::KERNEL[2] * hist[(i + 1) % rays];
            }

            // ── 3. Score each candidate direction ────────────────────── //
            int   best_ray   = 0;
            ftype best_score = -1e9f;

            for (int i = 0; i < rays; i++) {
                ftype delta = 2.0f * PI * i / (ftype)rays;
                if (delta > PI) delta -= 2.0f * PI;

                ftype score = -smooth[i]
                            + cfg::FORWARD_ALPHA * cos(delta)
                            - cfg::STEER_BETA    * abs(delta);

                if (score > best_score) {
                    best_score = score;
                    best_ray   = i;
                }
            }

            // ── 4. PD steering toward best direction ─────────────────── //
            ftype steer_err = 2.0f * PI * best_ray / (ftype)rays;
            if (steer_err > PI) steer_err -= 2.0f * PI;

            ftype d_err = steer_err - prev_steer_err;
            prev_steer_err = steer_err;

            steer = cfg::STEER_KP * steer_err + cfg::STEER_KD * d_err;

            // ── 5. Speed: average clearance in front +-60 cone ───────── //
            const int cone = rays / 6;
            ftype cone_dist = 0.0f;
            for (int i = -cone; i <= cone; i++)
                cone_dist += dist(mypos, raycasts[(i + rays) % rays]);
            cone_dist /= (ftype)(2 * cone + 1);

            ftype target_a;
            if (cone_dist > cfg::SAFE_DIST) {
                target_a = acceldelta * cfg::ACCEL_SCALE;
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

    // ------------------------------------------------------------------ //
    //  Simulation setup
    // ------------------------------------------------------------------ //
    array<agent, playercount> myagents;
    for (int i = 0; i < playercount; i++) myagents[i] = myagent;

    simulationinstance s(myagents, /*endtime=*/60.0f);

    // ------------------------------------------------------------------ //
    //  Obstacle movement specifiers
    //  [0 .. STATIC_COUNT-1]  → static
    //  [STATIC_COUNT .. N-2]  → Lissajous dynamic
    //  [N-1]                  → boundary wall (always static)
    // ------------------------------------------------------------------ //
    const int num_obs = (int)s.mp.size() - 1;

    vector<vector<point>> initial_obs(s.mp.begin(), s.mp.begin() + num_obs);

    auto no_movement = [](vector<point>&, const ftype) {};

    for (int i = 0; i < num_obs; i++) {
        if (i < cfg::STATIC_COUNT) {
            s.movementspecifier[i] = no_movement;
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