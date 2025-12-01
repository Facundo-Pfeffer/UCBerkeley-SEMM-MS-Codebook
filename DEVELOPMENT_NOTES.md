# Development Notes

This document contains important conventions and guidelines for maintaining this portfolio website.

## Path Conventions for HTML Files

### Important: GitHub Pages Deployment & Path Handling

The website is deployed via GitHub Actions (`.github/workflows/deploy-pages.yml`). The deployment process:
1. Copies all content from `webpage/` folder to `dist/` (the deployment root)
2. Copies all `highlighted_pdfs` folders from course directories **preserving their relative paths** into `dist/`

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

### Best Practice

When creating new PDF links or previews:
1. **Always check the deployment workflow** (`.github/workflows/deploy-pages.yml`)  
2. Copy the path format from working examples (CE220, CE225, CV)
3. **DO NOT use `../` prefix** - paths are relative to the deployed site root, not the repository structure
4. Test on the actual deployed GitHub Pages site, not just locally

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

*Last Updated: November 2025*
*Maintained by: Facundo L. Pfeffer*

