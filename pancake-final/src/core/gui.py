import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional

from src.core.state import PancakeState
from src.core.search import solve
from src.core.io import load_puzzle
from src.core.heuristics import get_heuristic, list_heuristics


class PancakeGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pancake Puzzle Solver")
        self.root.geometry("1000x700")
        self.root.minsize(900, 650)

        self.current_state: Optional[PancakeState] = None
        self.initial_state: Optional[PancakeState] = None
        self.current_file: Optional[Path] = None

        self.solution_states = []
        self.solution_moves = []
        self.playback_index = 0

        self.algorithm_var = tk.StringVar(value="A*")
        self.heuristic_var = tk.StringVar(value="gap")
        self.weight_var = tk.StringVar(value="1.5")
        self.status_var = tk.StringVar(value="Load a puzzle to begin.")
        self.moves_var = tk.StringVar(value="Manual moves: 0")

        self.stats_vars = {
            "solved": tk.StringVar(value="-"),
            "cost": tk.StringVar(value="-"),
            "expanded": tk.StringVar(value="-"),
            "generated": tk.StringVar(value="-"),
            "frontier": tk.StringVar(value="-"),
            "runtime": tk.StringVar(value="-"),
        }

        self.manual_move_count = 0

        self._build_ui()

    # ------------------------------------------------------------------
    # UI BUILD
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.grid(row=0, column=0, sticky="ew")
        top_frame.columnconfigure(8, weight=1)

        ttk.Button(top_frame, text="Load Puzzle", command=self.load_puzzle_file).grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Button(top_frame, text="Reset", command=self.reset_puzzle).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Button(top_frame, text="Solve", command=self.solve_current).grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Button(top_frame, text="Hint", command=self.show_hint).grid(
            row=0, column=3, padx=5, pady=5
        )
        ttk.Button(top_frame, text="Play Solution", command=self.play_solution).grid(
            row=0, column=4, padx=5, pady=5
        )

        ttk.Label(top_frame, text="Algorithm:").grid(row=0, column=5, padx=(20, 5))
        algorithm_combo = ttk.Combobox(
            top_frame,
            textvariable=self.algorithm_var,
            state="readonly",
            values=[
                "BFS",
                "DFS",
                "IDS",
                "UCS",
                "Greedy",
                "A*",
                "Weighted A*",
            ],
            width=14,
        )
        algorithm_combo.grid(row=0, column=6, padx=5)
        algorithm_combo.bind("<<ComboboxSelected>>", self._on_algorithm_change)

        ttk.Label(top_frame, text="Heuristic:").grid(row=0, column=7, padx=(20, 5))
        self.heuristic_combo = ttk.Combobox(
            top_frame,
            textvariable=self.heuristic_var,
            state="readonly",
            values=list_heuristics(),
            width=10,
        )
        self.heuristic_combo.grid(row=0, column=8, padx=5, sticky="w")

        ttk.Label(top_frame, text="Weight:").grid(row=0, column=9, padx=(20, 5))
        self.weight_entry = ttk.Entry(top_frame, textvariable=self.weight_var, width=8)
        self.weight_entry.grid(row=0, column=10, padx=5)

        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=1, column=0, sticky="nsew")
        main_frame.columnconfigure(0, weight=3)
        main_frame.columnconfigure(1, weight=2)
        main_frame.rowconfigure(0, weight=1)

        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_frame.rowconfigure(0, weight=1)
        left_frame.rowconfigure(1, weight=0)
        left_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left_frame, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _: self.draw_state())

        self.flip_buttons_frame = ttk.LabelFrame(left_frame, text="Manual Flip Controls", padding=10)
        self.flip_buttons_frame.grid(row=1, column=0, sticky="ew", pady=(10, 0))

        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky="nsew")
        right_frame.columnconfigure(0, weight=1)

        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding=10)
        stats_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        stats_frame.columnconfigure(1, weight=1)

        stats_labels = [
            ("Solved", "solved"),
            ("Solution Cost", "cost"),
            ("Nodes Expanded", "expanded"),
            ("Nodes Generated", "generated"),
            ("Max Frontier", "frontier"),
            ("Runtime (s)", "runtime"),
        ]

        for i, (label_text, key) in enumerate(stats_labels):
            ttk.Label(stats_frame, text=f"{label_text}:").grid(row=i, column=0, sticky="w", padx=5, pady=3)
            ttk.Label(stats_frame, textvariable=self.stats_vars[key]).grid(row=i, column=1, sticky="w", padx=5, pady=3)

        info_frame = ttk.LabelFrame(right_frame, text="Info", padding=10)
        info_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        info_frame.columnconfigure(0, weight=1)

        ttk.Label(info_frame, textvariable=self.moves_var).grid(row=0, column=0, sticky="w", pady=3)
        ttk.Label(
            info_frame,
            textvariable=self.status_var,
            wraplength=320,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=3)

        self.solution_box = tk.Text(right_frame, height=18, wrap="word", state="disabled")
        self.solution_box.grid(row=2, column=0, sticky="nsew")
        right_frame.rowconfigure(2, weight=1)

        self._on_algorithm_change()

    # ------------------------------------------------------------------
    # FILE / RESET
    # ------------------------------------------------------------------

    def load_puzzle_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Open Pancake Puzzle",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
        )
        if not file_path:
            return

        try:
            state = load_puzzle(file_path)
        except Exception as exc:
            messagebox.showerror("Load Error", f"Could not load puzzle:\n{exc}")
            return

        self.current_file = Path(file_path)
        self.initial_state = state
        self.current_state = state
        self.manual_move_count = 0
        self.solution_states = []
        self.solution_moves = []
        self.playback_index = 0

        self.moves_var.set("Manual moves: 0")
        self.status_var.set(f"Loaded puzzle: {self.current_file.name}")
        self._clear_stats()
        self._set_solution_text("")
        self.draw_state()
        self._refresh_flip_buttons()

    def reset_puzzle(self) -> None:
        if self.initial_state is None:
            return

        self.current_state = self.initial_state
        self.manual_move_count = 0
        self.solution_states = []
        self.solution_moves = []
        self.playback_index = 0

        self.moves_var.set("Manual moves: 0")
        self.status_var.set("Puzzle reset.")
        self._clear_stats()
        self._set_solution_text("")
        self.draw_state()
        self._refresh_flip_buttons()

    # ------------------------------------------------------------------
    # DRAWING
    # ------------------------------------------------------------------

    def draw_state(self) -> None:
        self.canvas.delete("all")

        if self.current_state is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2,
                self.canvas.winfo_height() // 2,
                text="Load a puzzle to display the pancake stack.",
                font=("Arial", 14),
            )
            return

        stack = self._extract_stack(self.current_state)
        if not stack:
            return

        canvas_width = max(self.canvas.winfo_width(), 400)
        canvas_height = max(self.canvas.winfo_height(), 400)

        n = len(stack)
        pancake_height = max(28, min(60, (canvas_height - 80) // max(1, n)))
        top_margin = 40
        center_x = canvas_width // 2
        max_width = min(500, canvas_width - 80)
        min_width = 100

        max_value = max(stack)

        self.canvas.create_text(
            center_x,
            20,
            text=f"Current Stack: {stack}",
            font=("Arial", 14, "bold"),
        )

        for i, value in enumerate(stack):
            width = min_width
            if max_value > 1:
                width = min_width + int((value - 1) * (max_width - min_width) / (max_value - 1))

            x1 = center_x - width // 2
            x2 = center_x + width // 2
            y1 = top_margin + i * pancake_height
            y2 = y1 + pancake_height - 6

            self.canvas.create_rectangle(x1, y1, x2, y2, outline="black", width=2, fill="#f0c674")
            self.canvas.create_text(center_x, (y1 + y2) // 2, text=str(value), font=("Arial", 12, "bold"))

    # ------------------------------------------------------------------
    # MANUAL PLAY
    # ------------------------------------------------------------------

    def _refresh_flip_buttons(self) -> None:
        for widget in self.flip_buttons_frame.winfo_children():
            widget.destroy()

        if self.current_state is None:
            ttk.Label(self.flip_buttons_frame, text="Load a puzzle first.").pack(anchor="w")
            return

        stack = self._extract_stack(self.current_state)
        ttk.Label(self.flip_buttons_frame, text="Choose a flip:").pack(anchor="w", pady=(0, 8))

        buttons_row = ttk.Frame(self.flip_buttons_frame)
        buttons_row.pack(anchor="w")

        for k in range(2, len(stack) + 1):
            ttk.Button(
                buttons_row,
                text=f"Flip {k}",
                command=lambda kk=k: self.apply_manual_flip(kk),
            ).pack(side="left", padx=4, pady=4)

    def apply_manual_flip(self, k: int) -> None:
        if self.current_state is None:
            return

        try:
            self.current_state = self.current_state.flip(k)
        except Exception as exc:
            messagebox.showerror("Flip Error", f"Could not apply flip {k}:\n{exc}")
            return

        self.manual_move_count += 1
        self.moves_var.set(f"Manual moves: {self.manual_move_count}")
        self.status_var.set(f"Applied manual flip({k}).")
        self.draw_state()

        if self.current_state.is_goal():
            messagebox.showinfo(
                "Solved",
                f"Puzzle solved manually in {self.manual_move_count} moves."
            )

    # ------------------------------------------------------------------
    # SOLVING / HINTS
    # ------------------------------------------------------------------

    def solve_current(self) -> None:
        if self.current_state is None:
            messagebox.showwarning("No Puzzle", "Load a puzzle first.")
            return

        try:
            result = self._run_solver(self.current_state)
        except Exception as exc:
            messagebox.showerror("Solve Error", f"Solver failed:\n{exc}")
            return

        self._update_stats(result)
        self._store_solution(result)
        self._show_solution_summary(result)

        if getattr(result, "solved", False):
            self.status_var.set("Solver found a solution.")
        else:
            self.status_var.set("Solver did not find a solution.")

    def show_hint(self) -> None:
        if self.current_state is None:
            messagebox.showwarning("No Puzzle", "Load a puzzle first.")
            return

        try:
            result = self._run_solver(self.current_state)
        except Exception as exc:
            messagebox.showerror("Hint Error", f"Could not compute hint:\n{exc}")
            return

        moves = getattr(result, "solution_moves", None) or getattr(result, "moves", None) or []
        if not getattr(result, "solved", False) or not moves:
            messagebox.showinfo("Hint", "No hint available from the current state.")
            return

        next_move = moves[0]
        messagebox.showinfo("Hint", f"Recommended next move: flip({next_move})")

    def play_solution(self) -> None:
        if not self.solution_states:
            messagebox.showinfo("No Solution", "Run the solver first.")
            return

        self.playback_index = 0
        self._play_next_state()

    def _play_next_state(self) -> None:
        if self.playback_index >= len(self.solution_states):
            return

        self.current_state = self.solution_states[self.playback_index]
        self.draw_state()
        self.playback_index += 1

        if self.playback_index < len(self.solution_states):
            self.root.after(500, self._play_next_state)

    def _run_solver(self, state: PancakeState):
        algorithm_display = self.algorithm_var.get()
        heuristic_name = self.heuristic_var.get()

        algorithm_map = {
            "BFS": "bfs",
            "DFS": "dfs",
            "IDS": "ids",
            "UCS": "ucs",
            "Greedy": "greedy",
            "A*": "astar",
            "Weighted A*": "weighted_astar",
        }

        algorithm = algorithm_map.get(algorithm_display)
        if algorithm is None:
            raise ValueError(f"Unsupported algorithm: {algorithm_display}")

        if algorithm == "weighted_astar":
            try:
                weight = float(self.weight_var.get())
            except ValueError as exc:
                raise ValueError("Weight must be a valid number.") from exc

            heuristic_fn = get_heuristic(heuristic_name)
            return solve(state, algorithm=algorithm, heuristic=heuristic_fn, weight=weight)

        if algorithm in {"greedy", "astar"}:
            heuristic_fn = get_heuristic(heuristic_name)
            return solve(state, algorithm=algorithm, heuristic=heuristic_fn)

        return solve(state, algorithm=algorithm)

    # ------------------------------------------------------------------
    # STATS / SOLUTION TEXT
    # ------------------------------------------------------------------

    def _update_stats(self, result) -> None:
        self.stats_vars["solved"].set(str(getattr(result, "solved", "-")))
        self.stats_vars["cost"].set(str(getattr(result, "solution_cost", "-")))
        self.stats_vars["expanded"].set(str(getattr(result, "nodes_expanded", "-")))
        self.stats_vars["generated"].set(str(getattr(result, "nodes_generated", "-")))
        self.stats_vars["frontier"].set(str(getattr(result, "max_frontier_size", "-")))

        runtime = getattr(result, "runtime_seconds", "-")
        if isinstance(runtime, float):
            runtime = f"{runtime:.6f}"
        self.stats_vars["runtime"].set(str(runtime))

    def _store_solution(self, result) -> None:
        self.solution_states = (
            getattr(result, "solution_states", None)
            or getattr(result, "states", None)
            or []
        )
        self.solution_moves = (
            getattr(result, "solution_moves", None)
            or getattr(result, "moves", None)
            or []
        )

    def _show_solution_summary(self, result) -> None:
        solved = getattr(result, "solved", False)
        moves = getattr(result, "solution_moves", None) or getattr(result, "moves", None) or []
        states = getattr(result, "solution_states", None) or getattr(result, "states", None) or []

        lines = []
        lines.append(f"Algorithm: {self.algorithm_var.get()}")
        lines.append(f"Heuristic: {self.heuristic_var.get() if self.algorithm_var.get() in {'Greedy', 'A*', 'Weighted A*'} else 'None'}")

        if self.algorithm_var.get() == "Weighted A*":
            lines.append(f"Weight: {self.weight_var.get()}")

        lines.append(f"Solved: {solved}")
        lines.append(f"Solution cost: {getattr(result, 'solution_cost', '-')}")
        lines.append(f"Moves: {moves}")
        lines.append("")

        if states:
            lines.append("State path:")
            for i, st in enumerate(states):
                try:
                    stack = self._extract_stack(st)
                except Exception:
                    stack = str(st)
                lines.append(f"{i}: {stack}")

        self._set_solution_text("\n".join(lines))

    def _set_solution_text(self, text: str) -> None:
        self.solution_box.config(state="normal")
        self.solution_box.delete("1.0", tk.END)
        self.solution_box.insert(tk.END, text)
        self.solution_box.config(state="disabled")

    def _clear_stats(self) -> None:
        for var in self.stats_vars.values():
            var.set("-")

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _on_algorithm_change(self, event=None) -> None:
        algorithm = self.algorithm_var.get()
        uses_heuristic = algorithm in {"Greedy", "A*", "Weighted A*"}
        uses_weight = algorithm == "Weighted A*"

        heuristic_state = "readonly" if uses_heuristic else "disabled"
        weight_state = "normal" if uses_weight else "disabled"

        self.heuristic_combo.configure(state=heuristic_state)
        self.weight_entry.configure(state=weight_state)

    @staticmethod
    def _extract_stack(state) -> list[int]:
        if hasattr(state, "stack"):
            return list(state.stack)
        if hasattr(state, "pancakes"):
            return list(state.pancakes)
        if isinstance(state, (list, tuple)):
            return list(state)
        raise ValueError("Could not extract pancake stack from state.")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    root = tk.Tk()
    app = PancakeGUI(root)
    app.run()


if __name__ == "__main__":
    main()