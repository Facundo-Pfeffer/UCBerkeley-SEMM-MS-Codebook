# Development Notes

This document contains important conventions and guidelines for maintaining this portfolio website.

## Path Conventions for HTML Files

### Important: GitHub Pages Deployment & Path Handling

The website is deployed via GitHub Actions (`.github/workflows/deploy-pages.yml`). The deployment process:
1. Copies all content from `webpage/` folder to `dist/` (the deployment root)
2. Copies all `highlighted_pdfs` folders from course directories **preserving their relative paths** into `dist/`
3. Copies all `highlighted_htmls` folders from course directories **preserving their relative paths** into `dist/`

**CRITICAL:** The `webpage/` folder contents are copied directly to `dist/` root, meaning:
- `webpage/assets/` → `dist/assets/`
- `webpage/images/` → `dist/images/`
- `webpage/*.html` → `dist/*.html`

**Result:** HTML files and PDF folders end up at the same level in `dist/`, so paths should **NOT** use `../` prefix.

### Correct Path Format

When linking to PDFs from HTML files in the `webpage/` folder:

```html
<!-- ✅ CORRECT (works after deployment) -->
<a href="CEE225_Dynamics/highlighted_pdfs/CE%20225%20HW5%20-%20Facundo%20L.%20Pfeffer.pdf">

<!-- ❌ INCORRECT (results in 404 errors on deployed site) -->
<a href="../CEE225_Dynamics/highlighted_pdfs/CE%20225%20HW5%20-%20Facundo%20L.%20Pfeffer.pdf">
```

### Examples by Course

- **From `webpage/cee225-dynamics.html` to Dynamics PDFs:**  
  `CEE225_Dynamics/highlighted_pdfs/filename.pdf`

- **From `webpage/cee220-analysis.html` to Analysis PDFs:**  
  `CEE220_StructuralAnalysis/highlighted_pdfs/filename.pdf`

- **From `webpage/cee231-mechanics.html` to Mechanics PDFs:**  
  `CEE231_SolidMechanics/highlighted_pdfs/filename.pdf`

### Path Conventions for `highlighted_htmls` Files

**CRITICAL:** HTML files in `highlighted_htmls` folders (e.g., `CEE225_Dynamics/highlighted_htmls/`) must use paths that account for the deployment structure.

**Deployment Structure:**
- `CEE225_Dynamics/highlighted_htmls/final_project_menu.html` → `dist/CEE225_Dynamics/highlighted_htmls/final_project_menu.html`
- `webpage/assets/css/main.css` → `dist/assets/css/main.css`
- `webpage/images/logo.png` → `dist/images/logo.png`

**Correct Paths from `highlighted_htmls` files:**

```html
<!-- ✅ CORRECT - CSS and JS paths -->
<link rel="stylesheet" href="../../assets/css/main.css" />
<script src="../../assets/js/jquery.min.js"></script>

<!-- ✅ CORRECT - Image paths -->
<img src="../../images/logo.png" alt="" />

<!-- ✅ CORRECT - Navigation links to main pages -->
<a href="../../cee225-dynamics.html">CEE225</a>
<a href="../../index.html">Home</a>

<!-- ❌ INCORRECT - DO NOT include 'webpage/' in paths -->
<link rel="stylesheet" href="../../webpage/assets/css/main.css" />
<img src="../../webpage/images/logo.png" alt="" />
<a href="../../webpage/cee225-dynamics.html">CEE225</a>
```

**Why:** The deployment copies `webpage/` contents to `dist/` root, so `webpage/assets` becomes `dist/assets`. From `dist/CEE225_Dynamics/highlighted_htmls/`, you go up two levels (`../../`) to reach `dist/`, then access `assets/` or `images/` directly.

**Files Affected:**
- All HTML files in `CEE225_Dynamics/highlighted_htmls/`
- All HTML files in `CEE231_SolidMechanics/highlighted_htmls/`
- Any other `highlighted_htmls` folders

### Best Practice

When creating new PDF links or previews:
1. **Always check the deployment workflow** (`.github/workflows/deploy-pages.yml`)  
2. Copy the path format from working examples (CE220, CE225, CV)
3. **DO NOT use `../` prefix** - paths are relative to the deployed site root, not the repository structure
4. Test on the actual deployed GitHub Pages site, not just locally

When creating HTML files in `highlighted_htmls` folders:
1. **Use `../../assets/` and `../../images/`** (NOT `../../webpage/assets/`)
2. **Use `../../*.html`** for links to main pages (NOT `../../webpage/*.html`)
3. **Verify paths match the deployment structure** where `webpage/` contents are copied to `dist/` root
4. **Check existing working files** (e.g., `final_project_menu.html`, `step1_mode_shapes.html`) for reference

## Code Style and Consistency

### UC Berkeley Branding
- Primary color: `#003262` (Berkeley Blue)
- Secondary color: `#FDB515` (California Gold)
- Use consistent color scheme across all pages

### Formatting Patterns

All course pages (CEE220, CEE225, CEE231) should maintain consistent:
- Boxed sections with `background:#f9fafb; border:1px solid #e5e7eb`
- Grid layouts using `display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr))`
- PDF preview sections with desktop and mobile views
- Button styling matching UC Berkeley colors

### List Formatting
- All list items should end with periods for consistency
- Maintain proper spacing with `margin-bottom` properties

## Communication Protocol

### When Implementing Changes

If a requested change appears inconsistent with existing files, formatting patterns, or established conventions:

1. **Point out the potential inconsistency** - Politely note what seems off
2. **Explain the existing pattern** - Reference what's currently in place
3. **Ask for clarification** - Confirm whether to match existing patterns or intentionally diverge
4. **Collaborate on resolution** - Work together to find the best approach

### Example Scenarios

- Adding a new feature with different styling → Check if it should match existing site styles
- Modifying paths → Verify the relative path structure is correct
- Changing section layouts → Ensure consistency across similar pages
- Adding new content → Maintain the established tone and formatting

This collaborative approach helps maintain code quality and prevents breaking existing functionality.

## Common Issues and Solutions

### 404 Errors on PDFs
**Problem:** PDF links return 404 errors  
**Solution:** Check that paths use `../` prefix from `webpage/` folder

### CSS/JS Not Loading in `highlighted_htmls` Files
**Problem:** HTML files in `highlighted_htmls` folders don't load CSS/JS, showing unstyled content  
**Root Cause:** Using incorrect paths like `../../webpage/assets/css/main.css` instead of `../../assets/css/main.css`  
**Solution:** 
- Remove `webpage/` from all asset paths in `highlighted_htmls` files
- Use `../../assets/` instead of `../../webpage/assets/`
- Use `../../images/` instead of `../../webpage/images/`
- Use `../../*.html` instead of `../../webpage/*.html` for navigation links
- **Remember:** `webpage/` contents are copied to `dist/` root during deployment

### Inconsistent Styling
**Problem:** New sections look different from existing ones  
**Solution:** Copy styling from similar sections on other course pages

### Mobile Responsiveness
**Problem:** Content doesn't display well on mobile  
**Solution:** Use the grid layout pattern with `repeat(auto-fit, minmax(...))`

## File Structure Reference

```
Repository Root/
├── webpage/                           # All HTML files here
│   ├── index.html
│   ├── about.html
│   ├── cee225-dynamics.html
│   ├── cee220-analysis.html
│   └── cee231-mechanics.html
├── CEE225_Dynamics/                   # Dynamics course files
│   └── highlighted_pdfs/              # PDF files for CEE225
├── CEE220_StructuralAnalysis/         # Analysis course files
│   └── highlighted_pdfs/              # PDF files for CEE220
└── CEE231_SolidMechanics/             # Mechanics course files
    └── highlighted_pdfs/              # PDF files for CEE231
```

From any HTML file in `webpage/`, use `../` to access folders at root level.

---

---

## Recent Fixes and Lessons Learned

### December 2025: Path Issues in `highlighted_htmls` Files

**Issue:** HTML files in `CEE225_Dynamics/highlighted_htmls/` (final_project_menu.html, step1_mode_shapes.html, step2_damping.html, step3_combined.html) were not loading CSS/JS correctly, resulting in unstyled pages.

**Root Cause:** Files were using paths like `../../webpage/assets/css/main.css`, but after deployment, `webpage/` contents are copied to `dist/` root, so `webpage/assets` becomes `dist/assets`. From `dist/CEE225_Dynamics/highlighted_htmls/`, the correct path is `../../assets/css/main.css` (without `webpage/`).

**Files Fixed:**
- `final_project_menu.html`
- `step1_mode_shapes.html`
- `step2_damping.html`
- `step3_combined.html`

**Prevention:** Always verify paths in `highlighted_htmls` files match the deployment structure. Use `../../assets/` not `../../webpage/assets/`.

### December 2025: Template CSS/JS Loading Issues in `highlighted_htmls` Files

**Issue:** Step 1 and Step 2 pages (`step1_mode_shapes.html`, `step2_damping.html`) were not rendering correctly, while Problem 3 and Problem 4 pages (`problem3_summary.html`, `problem4_summary.html`) rendered perfectly.

**Root Cause Analysis:**

**Problem 3 & 4 (Working):**
- Use **inline `<style>` tags** with all CSS embedded directly in HTML
- Do NOT depend on external CSS/JS files
- Use `<!DOCTYPE html>` (lowercase)
- Self-contained HTML with embedded styles
- Only depend on CDN resources (MathJax, polyfill)

**Step 1 & 2 (Not Working):**
- Use **Phantom template** from HTML5 UP
- Depend on external CSS: `../../assets/css/main.css`
- Depend on external JS: `../../assets/js/jquery.min.js`, `browser.min.js`, `breakpoints.min.js`, `util.js`, `main.js`
- Use `<!DOCTYPE HTML>` (uppercase)
- Require Phantom template CSS/JS to render correctly
- Template structure: `<div id="wrapper">`, `<header>`, `<nav>`, `<section class="tiles">`, etc.

**Key Finding:** Files that depend on external template CSS/JS are more prone to path/loading issues in `highlighted_htmls` folders. Files with inline styles work reliably.

**Best Practice for `highlighted_htmls` Files:**

1. **Prefer inline styles** (like Problem 3/4) for reliability:
   ```html
   <style>
   body { font-family: Arial, sans-serif; ... }
   h1 { color: #003262; ... }
   </style>
   ```

2. **If using templates** (like Phantom), ensure:
   - All paths use `../../assets/` not `../../webpage/assets/`
   - Verify CSS/JS files load correctly in browser console
   - Test on deployed site, not just locally

3. **CDN resources** (MathJax, Plotly) work fine - use full URLs:
   ```html
   <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
   ```

**Files Affected:**
- `step1_mode_shapes.html` - Uses Phantom template, depends on external CSS/JS
- `step2_damping.html` - Uses Phantom template, depends on external CSS/JS
- `problem3_summary.html` - Uses inline styles, works perfectly ✅
- `problem4_summary.html` - Uses inline styles, works perfectly ✅

---

*Last Updated: December 2025*
*Maintained by: Facundo L. Pfeffer*

