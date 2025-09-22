# Clock Tree Synthesizer

This tool implements a bottom-up Regular Grid Matching (RGM) algorithm for clock tree synthesis. It builds a clock tree with minimal skew, optimized wirelength, and manages capacitive load using buffers and branch insertion while satisfying design constraints.

## Usage

```bash
python cts.py <input_file> [visualize_output]
```

* `<input_file>`: Path to the sink coordinate input file.
* `[visualize_output]`: Optional. Set to `True` to enable tree visualization (default is `False`).

### Example

```bash
python cts.py example_input.txt True
```

## Input Files

* **Testcases** are provided in the `testcases/` folder.
* You can also generate additional input files using `gen_input.py`, which requires the number of sinks (`num_sinks`) as an input parameter:

```bash
python gen_input.py <num_sinks>
```

This will create a new input file with randomly generated sink coordinates.

## Output

* The synthesized clock tree data is written to `{input_filename}_output.txt`.
* The visualized clock tree is saved to `vis_{input_filename}.png`.

  * Sinks are marked as **blue**,
  * Branches as **brown**,
  * Buffers as **green**.

## Requirements

* Python 3.x
* `matplotlib` (required only if visualization is enabled)
