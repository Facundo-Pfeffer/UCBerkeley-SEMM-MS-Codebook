/**
 * Reusable Navigation System - Universal Path Resolution
 * 
 * This script provides a centralized, data-driven navigation system
 * that works from any directory level (webpage/, highlighted_htmls/, etc.)
 * 
 * Features:
 * - Automatic path resolution based on current page location
 * - Works from root, subdirectories, and nested folders
 * - Centralized navigation data - update once, applies everywhere
 * - Follows abstraction and best practices
 * 
 * Usage:
 * 1. Include this script: <script src="../../assets/js/navigation.js"></script> (adjust path as needed)
 * 2. Navigation initializes automatically
 * 3. To add new pages, update the navigationData object below
 */

// Navigation data structure - easily extensible
const navigationData = {
	main: [
		{ href: 'index.html', text: 'Home' },
		{ href: 'cv.html', text: 'Curriculum Vitae' }
	],
	dropdowns: [
		{
			label: 'UC Berkeley Work',
			items: [
				{ href: 'cee225-dynamics.html', text: 'CEE225 - Structural Dynamics' },
				{ href: 'cee220-analysis.html', text: 'CEE220 - Structural Analysis' },
				{ href: 'cee231-mechanics.html', text: 'CEE231 - Solid Mechanics' },
				{ href: 'cee244-structures.html', text: 'CEE244 - Reinforced Concrete Structures' }
			]
		},
		{
			label: 'Personal Projects',
			items: [
				{ href: 'https://facundo-pfeffer.github.io/ACSAHE.github.io/', text: 'ACSAHE' }
			]
		}
	],
	additional: [
		{ href: 'about.html', text: 'About' },
		{ href: 'contact.html', text: 'Contact' }
	]
};

// Mobile menu data (same structure, but flattened)
const mobileMenuData = [
	{ href: 'index.html', text: 'Home' },
	{ href: 'cv.html', text: 'Curriculum Vitae' },
	{ type: 'section', text: 'UC Berkeley Work' },
	{ href: 'cee225-dynamics.html', text: 'CEE225 - Structural Dynamics' },
	{ href: 'cee220-analysis.html', text: 'CEE220 - Structural Analysis' },
	{ href: 'cee231-mechanics.html', text: 'CEE231 - Solid Mechanics' },
	{ href: 'cee244-structures.html', text: 'CEE244 - Reinforced Concrete Structures' },
	{ type: 'section', text: 'Personal Projects' },
	{ href: 'https://facundo-pfeffer.github.io/ACSAHE.github.io/', text: 'ACSAHE' },
	{ href: 'about.html', text: 'About' },
	{ href: 'contact.html', text: 'Contact' }
];

/**
 * Calculate the relative path prefix needed to reach the webpage root
 * Based on the current page's location in the directory structure
 */
function getPathPrefix() {
	const currentPath = window.location.pathname;
	const pathDepth = currentPath.split('/').filter(segment => segment && !segment.endsWith('.html')).length;
	
	// If we're in a highlighted_htmls folder or similar subdirectory
	// We need to go up to reach the webpage root
	// Example: /CEE225_Dynamics/highlighted_htmls/page.html -> ../../ (2 levels up)
	
	// Count how many directory levels deep we are from the root
	// Root pages (webpage/*.html) = 0 levels
	// highlighted_htmls pages = 2 levels (CEE225_Dynamics/highlighted_htmls/)
	
	let levelsUp = 0;
	if (currentPath.includes('/highlighted_htmls/')) {
		// We're in a highlighted_htmls folder
		// Path structure: /CEE225_Dynamics/highlighted_htmls/page.html
		// Need to go up 2 levels to reach root
		levelsUp = 2;
	} else if (currentPath.includes('/highlighted_pdfs/')) {
		// Similar structure for PDF folders
		levelsUp = 2;
	} else if (currentPath.startsWith('/webpage/') || currentPath.match(/^\/[^\/]+\.html$/)) {
		// We're in the webpage root or at site root
		levelsUp = 0;
	} else {
		// Try to detect depth from path segments
		const segments = currentPath.split('/').filter(s => s && !s.endsWith('.html'));
		levelsUp = segments.length;
	}
	
	// Generate the prefix (../../ for 2 levels, ../ for 1 level, '' for root)
	return '../'.repeat(levelsUp);
}

/**
 * Resolve a path relative to the webpage root
 */
function resolvePath(href) {
	// External URLs don't need resolution
	if (href.startsWith('http://') || href.startsWith('https://') || href.startsWith('#')) {
		return href;
	}
	
	const prefix = getPathPrefix();
	return prefix + href;
}

/**
 * Generate desktop navigation HTML with proper path resolution
 */
function generateDesktopNav() {
	const prefix = getPathPrefix();
	let html = '<nav id="top-nav" class="desktop-nav"><ul>';
	
	// Main links
	navigationData.main.forEach(item => {
		html += `<li><a href="${resolvePath(item.href)}">${item.text}</a></li>`;
	});
	
	// Dropdowns
	navigationData.dropdowns.forEach(dropdown => {
		html += `<li class="dropdown">`;
		html += `<a href="#" class="dropdown-toggle">${dropdown.label} <span class="dropdown-arrow">▼</span></a>`;
		html += `<ul class="dropdown-menu">`;
		dropdown.items.forEach(item => {
			html += `<li><a href="${resolvePath(item.href)}">${item.text}</a></li>`;
		});
		html += `</ul></li>`;
	});
	
	// Additional links
	navigationData.additional.forEach(item => {
		html += `<li><a href="${resolvePath(item.href)}">${item.text}</a></li>`;
	});
	
	html += '</ul></nav>';
	return html;
}

/**
 * Generate mobile menu HTML with proper path resolution
 */
function generateMobileMenu() {
	let html = '<nav id="menu"><h2>Menu</h2><ul>';
	
	mobileMenuData.forEach(item => {
		if (item.type === 'section') {
			html += `<li><strong style="color: #003262; display: block; padding: 1em 0 0.5em 0; border-top: solid 1px rgba(255, 255, 255, 0.15); margin-top: 0.5em;">${item.text}</strong></li>`;
		} else {
			html += `<li><a href="${resolvePath(item.href)}">${item.text}</a></li>`;
		}
	});
	
	html += '</ul></nav>';
	return html;
}

/**
 * Initialize navigation on the page
 * Works from any directory level by automatically resolving paths
 */
function initNavigation() {
	// Find or create header
	let header = document.querySelector('#header .inner');
	if (!header) {
		console.warn('Navigation: Header not found, creating one');
		const wrapper = document.querySelector('#wrapper') || document.body;
		const headerEl = document.createElement('header');
		headerEl.id = 'header';
		headerEl.innerHTML = '<div class="inner"></div>';
		wrapper.insertBefore(headerEl, wrapper.firstChild);
		header = headerEl.querySelector('.inner');
	}
	
	// Ensure logo exists with proper path
	let logo = header.querySelector('.logo');
	const prefix = getPathPrefix();
	if (!logo) {
		logo = document.createElement('a');
		logo.href = resolvePath('index.html');
		logo.className = 'logo';
		logo.innerHTML = `<span class="symbol"><img src="${prefix}images/logo.png" alt="" /></span><span class="title">Facundo L. Pfeffer</span>`;
		header.insertBefore(logo, header.firstChild);
	} else {
		// Update logo paths if they exist
		const logoImg = logo.querySelector('img');
		if (logoImg && !logoImg.src.includes('http')) {
			const currentSrc = logoImg.getAttribute('src');
			if (!currentSrc.startsWith('../') && !currentSrc.startsWith('http')) {
				logoImg.src = prefix + 'images/logo.png';
			}
		}
		// Update logo link
		if (logo.href && !logo.href.includes('http') && !logo.href.startsWith('#')) {
			logo.href = resolvePath('index.html');
		}
	}
	
	// Replace or add desktop navigation
	const existingDesktopNav = header.querySelector('.desktop-nav');
	if (existingDesktopNav) {
		existingDesktopNav.outerHTML = generateDesktopNav();
	} else {
		const desktopNav = document.createElement('div');
		desktopNav.innerHTML = generateDesktopNav();
		header.appendChild(desktopNav.firstElementChild);
	}
	
	// Replace or add mobile menu button
	const existingMobileNav = header.querySelector('.mobile-nav');
	if (existingMobileNav) {
		existingMobileNav.outerHTML = '<nav class="mobile-nav"><ul><li><a href="#menu">Menu</a></li></ul></nav>';
	} else {
		// Remove old nav elements that aren't mobile-nav
		const oldNavs = header.querySelectorAll('nav:not(.desktop-nav):not(.mobile-nav)');
		oldNavs.forEach(nav => nav.remove());
		
		const mobileNav = document.createElement('nav');
		mobileNav.className = 'mobile-nav';
		mobileNav.innerHTML = '<ul><li><a href="#menu">Menu</a></li></ul>';
		header.appendChild(mobileNav);
	}
	
	// Replace or add mobile menu
	let mobileMenu = document.querySelector('#menu');
	const mobileMenuHTML = generateMobileMenu().replace('<nav id="menu">', '').replace('</nav>', '');
	if (mobileMenu) {
		mobileMenu.innerHTML = mobileMenuHTML;
	} else {
		mobileMenu = document.createElement('nav');
		mobileMenu.id = 'menu';
		mobileMenu.innerHTML = mobileMenuHTML;
		document.body.appendChild(mobileMenu);
	}
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
	document.addEventListener('DOMContentLoaded', initNavigation);
} else {
	initNavigation();
}

// Export for manual initialization if needed
if (typeof module !== 'undefined' && module.exports) {
	module.exports = { initNavigation, navigationData, mobileMenuData, getPathPrefix, resolvePath };
}
