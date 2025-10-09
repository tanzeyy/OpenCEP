# Module `scripts` 

This directory contains utility scripts and the main entry point for running the OpenCEP project.

## Structure of the Directory `scripts`

- `main.py`: Main script to run the application.
- `BikeTripUtils.py`: Contains utility classes and functions for bike trip data processing.
- `__init__.py`: Marks this directory as a Python package.

## 
We use the Kleene closure in OpenCEP, which is implemented in base/PatternStructure.py → class KleeneClosureOperator.

## Usage

Run the main script from the project root:

```bash
python scripts/main.py
```
`
```
# PATTERN SEQ (BikeTrip+ a[], BikeTrip b) -> Look for one or more (+) BikeTrip events (a[]), followed by another BikeTrip (b).
# WHERE a[i+1].bike = a[i].bike AND b.end in {7,8,9}
# AND a[last].bike = b.bike AND a[i+1].start = a[i].end
# WITHIN 1h
# RETURN (a[1].start, a[i].end, b.end)
```