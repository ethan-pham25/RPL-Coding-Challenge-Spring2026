# UCI Rocket Project Liquids

Spring 2026 recruitment coding challenge: implement a shared ring buffer in Python.

## What Students Work On

Your task is to replace the intentionally bad starter in `solution.py` with a correct implementation of:

- `RingSpec`
- `SharedRingBuffer`

The reference contract is defined by the official tests in `tests/official/`.

## Repo Layout

- `solution.py`: starter submission file. It is intentionally wrong.
- `tests/official/`: official grading tests.
- `tests/student/student_test_template.py`: example file you can copy to write your own tests.
- `score.py`: simple local scoring script.

## Getting Started

1. Create a virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the official score against your submission:

```bash
python score.py
```

## Student Workflow

- Edit `solution.py`.
- Leave `tests/official/` alone when you want a real score.
- Copy `tests/student/student_test_template.py` to something like `tests/student/test_my_solution.py`.
- Add your own cases there as you build.

To run your own tests too:

```bash
python score.py --include-student-tests
```

## Challenge Contract

Your implementation should preserve the public API used by the tests, including:

- shared-memory backed storage via `multiprocessing.shared_memory`
- per-reader positions and alive flags
- wrap-around read and write memory views
- simple byte-copy helpers
- correct cleanup and unlink behavior

If you change signatures or remove attributes the official tests will fail.

## GitHub

This repo includes a basic GitHub Actions workflow that smoke-tests the starter and scoring harness on pushes and pull requests.
