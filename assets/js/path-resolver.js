/**
 * ROBUST Path Resolution System
 * =============================
 * 
 * This module provides a centralized, robust path resolution system that:
 * 1. Automatically detects the environment (local dev vs GitHub Pages)
 * 2. Handles all path types: navigation links, CSS, JS, images
 * 3. Works from any directory level
 * 4. Has clear documentation and is maintainable
 * 
 * IMPORTANT: This is the SINGLE SOURCE OF TRUTH for path resolution.
 * All path-related code should use functions from this module.
 * 
 * Usage:
 *   import { resolveNavPath, resolveAssetPath, getBasePath } from './path-resolver.js';
 *   const homeLink = resolveNavPath('index.html');
 *   const cssPath = resolveAssetPath('assets/css/main.css');
 */

/**
 * Configuration Constants
 * DO NOT MODIFY THESE UNLESS THE REPOSITORY STRUCTURE CHANGES
 */
const CONFIG = {
	// GitHub Pages repository subdirectory
	GITHUB_PAGES_BASE: '/UCBerkeley-SEMM-MS-Codebook/',
	
	// Local development: webpage folder is the root
	LOCAL_WEBPAGE_ROOT: 'webpage/',
	
	// Asset folders (relative to webpage root)
	ASSET_FOLDERS: {
		css: 'assets/css/',
		js: 'assets/js/',
		images: 'images/'
	}
};

/**
 * Environment Detection
 * ====================
 * Detects if we're running on GitHub Pages or local development
 */
function detectEnvironment() {
	const hostname = window.location.hostname;
	const pathname = window.location.pathname;
	
	// GitHub Pages detection
	if (hostname.includes('github.io') && pathname.includes(CONFIG.GITHUB_PAGES_BASE)) {
		return 'github-pages';
	}
	
	// Local development
	return 'local';
}

/**
 * Get Base Path
 * =============
 * Returns the base path for the current environment
 * 
 * @returns {string} Base path (e.g., '/UCBerkeley-SEMM-MS-Codebook/' or '')
 */
function getBasePath() {
	const env = detectEnvironment();
	
	if (env === 'github-pages') {
		return CONFIG.GITHUB_PAGES_BASE;
	}
	
	// Local: no base path needed (webpage/ is root)
	return '';
}

/**
 * Get Relative Path Prefix
 * ========================
 * Calculates relative path prefix needed to reach webpage root from current location
 * Only used for local development
 * 
 * @returns {string} Relative path prefix (e.g., '../../' or '')
 */
function getRelativePathPrefix() {
	const env = detectEnvironment();
	
	// On GitHub Pages, always use absolute paths
	if (env === 'github-pages') {
		return '';
	}
	
	// Local development: calculate relative path
	const currentPath = window.location.pathname;
	
	// Count directory levels from webpage root
	let levelsUp = 0;
	
	if (currentPath.includes('/highlighted_htmls/')) {
		levelsUp = 2; // e.g., /CEE225_Dynamics/highlighted_htmls/page.html
	} else if (currentPath.includes('/highlighted_pdfs/')) {
		levelsUp = 2; // e.g., /CEE225_Dynamics/highlighted_pdfs/file.pdf
	} else if (currentPath.startsWith('/webpage/') || currentPath.match(/^\/[^\/]+\.html$/)) {
		levelsUp = 0; // Already at webpage root
	} else {
		// Try to detect depth from path segments
		const segments = currentPath.split('/').filter(s => s && !s.endsWith('.html') && !s.endsWith('.pdf'));
		levelsUp = segments.length;
	}
	
	return '../'.repeat(levelsUp);
}

/**
 * Resolve Navigation Path
 * =======================
 * Resolves paths for navigation links (HTML pages)
 * 
 * @param {string} href - Relative path to HTML file (e.g., 'index.html', 'cv.html')
 * @returns {string} Resolved path
 */
function resolveNavPath(href) {
	// External URLs and anchors don't need resolution
	if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('#')) {
		return href;
	}
	
	const env = detectEnvironment();
	
	if (env === 'github-pages') {
		// Use absolute path from GitHub Pages root
		return CONFIG.GITHUB_PAGES_BASE + href;
	}
	
	// Local: use relative path
	const prefix = getRelativePathPrefix();
	return prefix + href;
}

/**
 * Resolve Asset Path
 * ==================
 * Resolves paths for assets (CSS, JS, images)
 * 
 * @param {string} assetPath - Relative path to asset (e.g., 'assets/css/main.css')
 * @returns {string} Resolved path
 */
function resolveAssetPath(assetPath) {
	// External URLs don't need resolution
	if (assetPath.startsWith('http://') || assetPath.startsWith('https://')) {
		return assetPath;
	}
	
	const env = detectEnvironment();
	
	if (env === 'github-pages') {
		// Use absolute path from GitHub Pages root
		return CONFIG.GITHUB_PAGES_BASE + assetPath;
	}
	
	// Local: use relative path
	const prefix = getRelativePathPrefix();
	return prefix + assetPath;
}

/**
 * Resolve Image Path
 * ==================
 * Convenience function for image paths
 * 
 * @param {string} imagePath - Relative path to image (e.g., 'images/logo.png')
 * @returns {string} Resolved path
 */
function resolveImagePath(imagePath) {
	return resolveAssetPath(imagePath);
}

// Export functions for use in other modules
if (typeof module !== 'undefined' && module.exports) {
	module.exports = {
		detectEnvironment,
		getBasePath,
		getRelativePathPrefix,
		resolveNavPath,
		resolveAssetPath,
		resolveImagePath,
		CONFIG
	};
}

// Make functions globally available for navigation.js
window.PathResolver = {
	detectEnvironment,
	getBasePath,
	getRelativePathPrefix,
	resolveNavPath,
	resolveAssetPath,
	resolveImagePath,
	CONFIG
};

