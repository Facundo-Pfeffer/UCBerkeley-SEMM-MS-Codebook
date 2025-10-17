# CEE231 – Solid Mechanics

This folder contains coursework and exploratory analyses for UC Berkeley's CEE231 (Solid Mechanics). It includes code to compute and visualize material response and a set of exported interactive HTML results.

## Contents

- `highlighted_htmls/` – Interactive HTML results published to the website automatically
  - `Directional_Youngs_Modulus_A.html` – Fe (cubic) directional Young’s modulus point cloud
  - `Directional_Youngs_Modulus_B.html` – Nb (cubic) directional Young’s modulus point cloud
  - `Directional_Youngs_Modulus_C.html` – NiTi alloy (general anisotropy) directional Young’s modulus point cloud
- `ce231_specific_homework_runs/` – scripts for assignments (`01HW.py` … `06HW.py`)

## How interactive results are published

The website is built from `webpage/` via GitHub Actions. During deployment, any folder named `highlighted_htmls` is mirrored into the published site at the same relative path. That means the contents of this directory are available at:

```
/CEE231_SolidMechanics/highlighted_htmls/
```

Concrete URLs (replace `<repo>` with your repository name):

```
https://<username>.github.io/<repo>/CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_A.html
https://<username>.github.io/<repo>/CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_B.html
https://<username>.github.io/<repo>/CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_C.html
```

No manual copying is needed—add or update files inside `highlighted_htmls/` and push to the default branch (or manually run the Pages workflow). The deploy action takes care of publishing them.

## Extend the results

1. Generate new interactive visualizations (e.g., directional Young’s modulus, compliance plots).
2. Export as self‑contained HTML and save them inside `highlighted_htmls/`.
3. Push to the repo. After deployment, the new files will be available under the same path on the website.

## Link from the website

From a page in `webpage/`, link using a repo‑relative path so it works both locally and on GitHub Pages. Example:

```html
<a href="CEE231_SolidMechanics/highlighted_htmls/Directional_Youngs_Modulus_A.html" target="_blank">Fe — Directional E (interactive)</a>
```

## Notes

- Prefer URL‑safe filenames (avoid spaces or URL‑encode them when linking).
- Keep large binaries out of git when possible; export optimized HTML for lighter downloads.
