# Statically Determinate Trusses

This project implements a solver for statically determinate planar trusses in Python.  
The program builds equilibrium matrices, solves for internal forces, and computes support reactions based on truss data defined in a JSON file.

---

## Functionality

The solver performs the following steps:

1. Reads the truss definition from a JSON file.
2. Creates `Node`, `Element`, and `FreeDOF` objects.
3. Constructs the equilibrium matrix for free DOFs (\( \bs{B}_f \)) and restrained DOFs (\( \bs{B}_d \)).
4. Solves for the internal element forces:
   \[
   \bs{Q} = \bs{B}_f^{-1}\bs{P}_f
   \]
5. Computes the support reactions:
   \[
   \bs{R} = \bs{B}_d\bs{Q}
   \]
6. Verifies global equilibrium (\(\Sigma F_x = 0\), \(\Sigma F_y = 0\), \(\Sigma M = 0\)).
7. Optionally produces a Plotly visualization of the truss geometry.

---

## JSON Input Format

The solver relies on a single JSON input file that specifies:

- **Nodes**  
  Each node requires coordinates and optional support conditions:
  ```json
  {"id": 1, "x": 0, "y": 0, "x_restrained": true, "y_restrained": true}
  ```
  - `id`: integer node identifier.  
  - `x`, `y`: coordinates.  
  - `x_restrained`, `y_restrained`: booleans defining supports (default is `false`).  

- **Elements**  
  Each element is a pair of node IDs:
  ```json
  [1, 2]
  ```
  This example creates a bar element between node 1 and node 2.

- **Loads**  
  Each load is defined by node, direction, and value:
  ```json
  {"node": 6, "direction": 2, "value": -10}
  ```
  - `node`: node ID where the load is applied.  
  - `direction`: `1` for x-direction, `2` for y-direction.  
  - `value`: load magnitude (positive right/up, negative left/down).  

---

## Example Input

```json
{
  "nodes": [
    {"id": 1, "x": 0, "y": 0, "x_restrained": true, "y_restrained": true},
    {"id": 2, "x": 8, "y": 0},
    {"id": 3, "x": 16, "y": 0, "y_restrained": true},
    {"id": 4, "x": 0, "y": 4},
    {"id": 5, "x": 16, "y": 4},
    {"id": 6, "x": 4, "y": 12},
    {"id": 7, "x": 12, "y": 12}
  ],
  "elements": [
    [1, 2], [2, 3], [1, 4], [3, 5], [4, 6], [5, 7],
    [6, 7], [4, 2], [2, 5], [3, 6], [1, 7]
  ],
  "loads": [
    {"node": 6, "direction": 2, "value": -10},
    {"node": 7, "direction": 2, "value": -15}
  ]
}
```

---

## Running the Solver

From the project directory, run:

```bash
python main.py HW1/input_json_files/HW1.json
```

This will:

- Print matrices (\( \bs{B}_f \), \( \bs{B}_f^{-1} \), \( \bs{B}_d \))  
- Output vectors (\( \bs{P}_f \), \( \bs{Q} \), \( \bs{R} \))  
- Verify global equilibrium  

---

## Reference

Equilibrium equations are derived following:

Filippou, F. C. *Structural Analysis: Theory and Applications*.  
University of California, Berkeley.  

One equilibrium equation is written per model DOF at a time, consistent with this reference.
