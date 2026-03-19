"""
Manim animation of the SIRD epidemic model.

Usage:
    uv run manim -pql animate_sird.py SIRDAnimation     # low quality preview
    uv run manim -pqh animate_sird.py SIRDAnimation     # high quality
"""

import numpy as np
from scipy.integrate import solve_ivp
from manim import (
    Scene, Axes, VGroup, Dot, MathTex, Text, Line, Rectangle,
    ValueTracker, always_redraw, Write, Create, FadeIn,
    ORIGIN, UP, DOWN, LEFT, RIGHT, WHITE, GRAY,
    rate_functions, config,
)
from manim import GREEN, ORANGE, RED, BLUE, DARK_GRAY

config.background_color = "#0f0f0f"

# ── Palette ───────────────────────────────────────────────────────────────────
C_S = GREEN
C_I = RED
C_R = BLUE
C_D = DARK_GRAY

# ── Solve SIRD upfront ────────────────────────────────────────────────────────
R0                = 2.5
infectious_period = 14
mortality_rate    = 0.005
N                 = 10_000
days              = 365
I0                = 1

gamma = 1 / infectious_period
beta  = R0 * (gamma + mortality_rate)
mu    = mortality_rate
S0    = (N - I0) / N


def sird_ode(t, y):
    S, I, R, D = y
    return [
        -beta * S * I,
         beta * S * I - gamma * I - mu * I,
         gamma * I,
         mu * I,
    ]


sol = solve_ivp(
    sird_ode, (0, days), [S0, I0 / N, 0.0, 0.0],
    t_eval=np.linspace(0, days, days * 4),
    method="RK45", rtol=1e-6,
)
t_arr      = sol.t
S_arr, I_arr, R_arr, D_arr = sol.y * 100


# ── Scene ─────────────────────────────────────────────────────────────────────
class SIRDAnimation(Scene):
    def construct(self):
        # Title
        title = Text("SIRD Epidemic Model", font_size=36, color=WHITE).to_edge(UP, buff=0.3)
        subtitle = Text(
            f"R₀={R0}  |  1/γ={infectious_period}d  |  μ={mu:.3f}/day  |  N={N:,}",
            font_size=18, color=GRAY,
        ).next_to(title, DOWN, buff=0.1)
        self.play(Write(title), FadeIn(subtitle))

        # Axes
        axes = Axes(
            x_range=[0, days, 60],
            y_range=[0, 105, 20],
            x_length=10,
            y_length=5,
            axis_config={"color": GRAY, "include_tip": False},
            x_axis_config={"numbers_to_include": np.arange(0, days + 1, 60)},
            y_axis_config={"numbers_to_include": np.arange(0, 101, 20)},
        ).shift(DOWN * 0.6)

        x_label = axes.get_x_axis_label(Text("Days", font_size=18, color=GRAY), direction=DOWN)
        y_label = axes.get_y_axis_label(Text("% population", font_size=18, color=GRAY), direction=LEFT)

        self.play(Create(axes), FadeIn(x_label), FadeIn(y_label))

        # Legend
        legend_items = [
            (C_S, "Susceptible"),
            (C_I, "Infectious"),
            (C_R, "Recovered"),
            (C_D, "Dead"),
        ]
        legend = VGroup()
        for color, label in legend_items:
            dot  = Dot(color=color, radius=0.08)
            text = Text(label, font_size=16, color=color)
            row  = VGroup(dot, text).arrange(RIGHT, buff=0.15)
            legend.add(row)
        legend.arrange(DOWN, aligned_edge=LEFT, buff=0.2).to_corner(RIGHT + UP, buff=0.8).shift(DOWN * 1.2)
        self.play(FadeIn(legend))

        # Tracker drives how far along the timeline we are
        tracker = ValueTracker(0)

        def make_curve(arr, color):
            def updater():
                idx = int(tracker.get_value())
                idx = max(1, min(idx, len(t_arr) - 1))
                points = [
                    axes.c2p(t_arr[i], arr[i])
                    for i in range(idx)
                ]
                curve = Line(points[0], points[0], color=color)
                if len(points) >= 2:
                    from manim import VMobject
                    curve = VMobject(color=color, stroke_width=2.5)
                    curve.set_points_smoothly(points)
                return curve
            return always_redraw(updater)

        curve_S = make_curve(S_arr, C_S)
        curve_I = make_curve(I_arr, C_I)
        curve_R = make_curve(R_arr, C_R)
        curve_D = make_curve(D_arr, C_D)

        # Day counter
        day_text = always_redraw(
            lambda: Text(
                f"Day {int(tracker.get_value() / 4):>3}",
                font_size=22, color=WHITE,
            ).to_corner(LEFT + UP, buff=0.8).shift(DOWN * 1.0)
        )

        self.add(curve_S, curve_I, curve_R, curve_D, day_text)

        # Animate
        self.play(
            tracker.animate.set_value(len(t_arr) - 1),
            run_time=12,
            rate_func=rate_functions.linear,
        )

        # Final annotations — peak infectious
        peak_idx = int(np.argmax(I_arr))
        peak_dot = Dot(axes.c2p(t_arr[peak_idx], I_arr[peak_idx]), color=C_I, radius=0.1)
        peak_label = Text(
            f"Peak {I_arr[peak_idx]:.1f}%\nday {t_arr[peak_idx]:.0f}",
            font_size=16, color=C_I,
        ).next_to(peak_dot, UP + RIGHT, buff=0.15)

        self.play(Create(peak_dot), Write(peak_label))
        self.wait(2)
