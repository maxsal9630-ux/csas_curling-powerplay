CSAS 2026 Data Challenge - Power Play Analysis (Mixed Doubles Curling)

This project analyzes how teams use the power play in mixed doubles curling and what happens when they do. The focus is on timing (when the power play is used) and execution (what teams do early in the end and how that relates to scoring).

Data
----
This project uses only the data provided by the Data Challenge:
Competition, Competitors, Teams, Games, Ends, and Stones (Curlit data).

The raw data files are not included here, per the submission instructions.
They should be placed in:
data/raw/

Code
----
code/load_data.py
- Loads the raw CSV files and prints basic structure and counts

code/build_end_table.py
- Builds a team-by-end table with score context, power play usage,
  score difference before the end, and end results

code/timing_analysis.py
- Power play timing analysis
- Compares power play vs non-power play outcomes by end and score state

code/powerplay_execution.py
- Shot selection and execution analysis
- Examines opening power play shots and how shot quality relates to scoring

code/team_benchmark.py
- Team-level benchmarking of power play effectiveness
- Compares average power play lift across teams

code/ctable.py
- Combines timing and outcome information into a single decision table
- Adds simple recommendation flags based on observed results

code/bench.py
- Helper script used for exploratory checks and intermediate testing

Analysis Scope
--------------
This project addresses several of the student deliverables, including:
- Power play timing optimization using score state and end number
- Shot selection probability summaries for opening power play shots
- Team-specific strategy recommendations based on observed power play outcomes
- Performance benchmarking across international teams

Outputs
-------
outputs/
team_end_table.csv
- Main analysis table at the team-end level

pp_usage_rate_by_state.csv
- How often teams use the power play by end and score situation

timing_summary.csv
- End outcomes with and without the power play

timing_lift.csv
- Difference in outcomes between power play and non-power play

decision_table.csv
- Combined table with usage, outcome differences, and recommendation flags

pp_opening_task_rates.csv
pp_opening_task_outcomes.csv
pp_points_quality_outcomes.csv
- Shot-level summaries for power play execution

team_benchmark.csv
- Team-level power play performance comparisons

Figures
-------
figures/
lift_points_tied.png
pp_shot1_task_mean_points.png
pp_shot1_task_p2plus.png
team_benchmark_lift_points.png

Running the Code
----------------
Place the given data files in:
data/raw/

Run the scripts in this order:
python3 code/load_data.py
python3 code/build_end_table.py
python3 code/timing_analysis.py
python3 code/powerplay_execution.py
python3 code/team_benchmark.py
python3 code/ctable.py

All tables and figures will be generated automatically.

Output:
- CSV files in outputs/ contain all summary tables used in the analysis.
- Figures in figures/ visualize key results related to power play timing,
  execution, and team performance.

If something looks surprising, thats probably the point...
