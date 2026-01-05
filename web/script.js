// Simulated news headlines for the feed
const newsHeadlines = [
    "Global markets rally as central banks signal rate cuts ahead",
    "AI breakthrough: New model achieves human-level reasoning on complex tasks",
    "Climate summit reaches historic agreement on carbon emissions",
    "Tech giants announce joint initiative for responsible AI development",
    "Breakthrough in quantum computing promises faster drug discovery",
    "Space agency confirms water ice deposits on lunar south pole",
    "Major earthquake strikes Pacific region, triggering tsunami warnings",
    "Revolutionary battery technology could double electric vehicle range",
    "Scientists discover high possibility high life signs on distant exoplanet",
    "Cybersecurity experts warn of sophisticated new ransomware variant",
    "Historic peace agreement signed after decades of regional conflict",
    "New study links gut microbiome to mental health outcomes",
    "Autonomous vehicles approved for highway use in three more states",
    "Central bank raises interest rates citing persistent inflation concerns",
    "Rare solar storm disrupts satellite communications worldwide",
    "Breakthrough gene therapy shows promise for treating rare diseases",
    "Major tech company announces largest layoffs in company history",
    "Archaeological discovery rewrites timeline of human migration",
    "New renewable energy project to power entire metropolitan area",
    "Scientists achieve nuclear fusion milestone at research facility"
];

// State
let state = {
    metaSurprise: 0,
    confidence: 0.88,
    salience: 0,
    arousal: 0,
    episodicCount: 0,
    semanticCount: 0,
    currentPhase: 0,
    feedItems: [],
    memories: []
};

const phases = ['predicting', 'experiencing', 'surprising', 'committing'];

// DOM elements
let elements = {};

document.addEventListener('DOMContentLoaded', () => {
    // Cache DOM elements
    elements = {
        feed: document.getElementById('feed'),
        metaSurprise: document.getElementById('metaSurprise'),
        metaSurpriseBar: document.getElementById('metaSurpriseBar'),
        confidence: document.getElementById('confidence'),
        confidenceBar: document.getElementById('confidenceBar'),
        salience: document.getElementById('salience'),
        salienceBar: document.getElementById('salienceBar'),
        arousal: document.getElementById('arousal'),
        arousalBar: document.getElementById('arousalBar'),
        episodicCount: document.getElementById('episodicCount'),
        semanticCount: document.getElementById('semanticCount'),
        currentPhase: document.getElementById('currentPhase'),
        memoryTrace: document.getElementById('memoryTrace'),
        centerDot: document.getElementById('centerDot'),
        loopLabels: document.querySelectorAll('.loop-label'),
        heroLogo: document.getElementById('heroLogo'),
        geoCircle: document.querySelector('.geo-circle'),
        header: document.querySelector('.site-header'),
        hero: document.querySelector('.hero')
    };

    // Initialize scroll-based circle animation
    initScrollAnimation();

    // Initialize hero logo animations
    initStarfieldAnimation();

    // Start the simulation
    startSimulation();

    // Console easter egg
    console.log('%c\u25C9 entitic.ai', 'font-size: 24px; font-weight: bold; color: #818cf8;');
    console.log('%cMemory Streams \u2014 Four temporal scales. One experiencing self.', 'color: #5a6270;');
    console.log('%c"You are here to develop yourself. Wake up and start with it."', 'color: #9ba1ab; font-style: italic;');
});

function initScrollAnimation() {
    const circle = elements.geoCircle;
    const hero = elements.hero;

    if (!circle || !hero) return;

    // Store initial position
    let initialRect = circle.getBoundingClientRect();
    let initialTop = initialRect.top + window.scrollY;
    let initialLeft = initialRect.left;
    let initialWidth = initialRect.width;

    // Target position (top left corner)
    const targetTop = 24;
    const targetLeft = 24;
    const targetScale = 0.25;

    let ticking = false;

    function updateCirclePosition() {
        const scrollY = window.scrollY;
        const heroHeight = hero.offsetHeight;
        const triggerPoint = heroHeight * 0.2;
        const endPoint = heroHeight * 0.6;

        // Calculate progress (0 = top, 1 = scrolled past trigger)
        let progress = 0;
        if (scrollY > triggerPoint) {
            progress = Math.min((scrollY - triggerPoint) / (endPoint - triggerPoint), 1);
        }

        // Ease the progress
        const easedProgress = easeOutCubic(progress);

        if (progress > 0) {
            // Calculate current position
            const currentTop = initialTop - scrollY;
            const currentLeft = initialLeft;

            // Calculate translation needed
            const translateX = (targetLeft - currentLeft) * easedProgress;
            const translateY = (targetTop - currentTop) * easedProgress;
            const scale = 1 - (1 - targetScale) * easedProgress;

            circle.style.position = 'fixed';
            circle.style.top = `${currentTop + translateY}px`;
            circle.style.left = `${currentLeft + translateX}px`;
            circle.style.transform = `scale(${scale})`;
            circle.style.transformOrigin = 'top left';
            circle.style.zIndex = '101';
        } else {
            // Reset to normal flow
            circle.style.position = '';
            circle.style.top = '';
            circle.style.left = '';
            circle.style.transform = '';
            circle.style.transformOrigin = '';
            circle.style.zIndex = '';
        }

        ticking = false;
    }

    function easeOutCubic(t) {
        return 1 - Math.pow(1 - t, 3);
    }

    // Recalculate initial position on resize
    window.addEventListener('resize', () => {
        if (window.scrollY === 0) {
            initialRect = circle.getBoundingClientRect();
            initialTop = initialRect.top + window.scrollY;
            initialLeft = initialRect.left;
            initialWidth = initialRect.width;
        }
    });

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(updateCirclePosition);
            ticking = true;
        }
    });

    // Initial call
    updateCirclePosition();
}

function initStarfieldAnimation() {
    if (!elements.heroLogo) return;

    const svg = elements.heroLogo.querySelector('svg');
    if (!svg) return;

    // Get all stars in the starfield
    const stars = svg.querySelectorAll('.starfield circle');
    const centerX = 200;
    const centerY = 200;
    const radius = 160;

    // Store original positions and convert to spherical coordinates
    const starData = [];
    stars.forEach((star) => {
        const cx = parseFloat(star.getAttribute('cx'));
        const cy = parseFloat(star.getAttribute('cy'));
        const baseOpacity = parseFloat(star.getAttribute('opacity')) || 0.5;
        const baseRadius = parseFloat(star.getAttribute('r')) || 2;

        // Convert 2D position to spherical coordinates
        const dx = cx - centerX;
        const dy = cy - centerY;
        const distFromCenter = Math.sqrt(dx * dx + dy * dy);

        // Map to sphere: theta (horizontal angle), phi (vertical angle)
        const theta = Math.atan2(dy, dx);
        const phi = Math.acos(Math.min(1, distFromCenter / radius));

        starData.push({
            element: star,
            theta: theta,
            phi: phi,
            baseOpacity: baseOpacity,
            baseRadius: baseRadius,
            originalR: baseRadius
        });
    });

    // Animation loop for sphere rotation
    let rotationY = 0;
    let rotationX = 0;

    function animateSphere() {
        rotationY += 0.003; // Slow horizontal rotation
        rotationX += 0.001; // Very slow vertical wobble

        starData.forEach((data) => {
            // Apply rotation to spherical coordinates
            const newTheta = data.theta + rotationY;
            const newPhi = data.phi + Math.sin(rotationX) * 0.1;

            // Convert back to 3D cartesian
            const x = radius * Math.sin(newPhi) * Math.cos(newTheta);
            const y = radius * Math.sin(newPhi) * Math.sin(newTheta);
            const z = radius * Math.cos(newPhi);

            // Project to 2D with perspective
            const perspective = 400;
            const scale = perspective / (perspective + z);

            const screenX = centerX + x * scale;
            const screenY = centerY + y * scale;

            // Update position
            data.element.setAttribute('cx', screenX);
            data.element.setAttribute('cy', screenY);

            // Adjust size and opacity based on z-depth (closer = bigger/brighter)
            const depthScale = (z + radius) / (2 * radius); // 0 to 1, front to back
            const newOpacity = data.baseOpacity * (0.3 + depthScale * 0.7);
            const newRadius = data.baseRadius * (0.6 + depthScale * 0.5);

            data.element.setAttribute('opacity', newOpacity);
            data.element.setAttribute('r', newRadius);
        });

        requestAnimationFrame(animateSphere);
    }

    animateSphere();

    // Add CSS for boundary rotation
    if (!document.getElementById('starfield-styles')) {
        const style = document.createElement('style');
        style.id = 'starfield-styles';
        style.textContent = `
            .circle-boundary {
                animation: slowRotate 120s linear infinite;
                transform-origin: 200px 200px;
            }

            @keyframes slowRotate {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
        `;
        document.head.appendChild(style);
    }
}

function startSimulation() {
    // Add initial feed items
    for (let i = 0; i < 3; i++) {
        setTimeout(() => addFeedItem(), i * 500);
    }

    // Continuous feed
    setInterval(addFeedItem, 4000);

    // Update metrics continuously
    setInterval(updateMetrics, 100);

    // Cycle through phases
    setInterval(cyclePhase, 1000);
}

function addFeedItem() {
    const headline = newsHeadlines[Math.floor(Math.random() * newsHeadlines.length)];
    const timestamp = new Date().toLocaleTimeString('en-US', { hour12: false });

    // Calculate simulated metrics for this item
    const surprise = 0.3 + Math.random() * 0.7;
    const itemArousal = 0.2 + Math.random() * 0.6;
    const itemSalience = surprise * itemArousal * (1 + Math.random() * 0.5);
    const shouldCrystallize = itemSalience > 0.4;

    // Create feed item
    const item = document.createElement('div');
    item.className = 'feed-item processing';
    item.innerHTML = `
        <div class="feed-item-text">${headline}</div>
        <div class="feed-item-meta">
            <span class="feed-item-time">${timestamp}</span>
            <span class="feed-item-tag">processing</span>
        </div>
    `;

    // Add to feed
    elements.feed.insertBefore(item, elements.feed.firstChild);

    // Limit feed items
    while (elements.feed.children.length > 15) {
        elements.feed.removeChild(elements.feed.lastChild);
    }

    // Update state
    state.metaSurprise = 0.3 + Math.random() * 0.4;
    state.arousal = itemArousal;
    state.salience = itemSalience;

    // Animate processing
    setTimeout(() => {
        item.classList.remove('processing');

        if (shouldCrystallize) {
            item.classList.add('crystallized');
            item.querySelector('.feed-item-tag').textContent = 'crystallized';
            crystallizeMemory(headline, itemSalience);
        } else {
            item.querySelector('.feed-item-tag').textContent = 'processed';
        }

        // Update confidence based on meta-surprise (inverse relationship)
        state.confidence = 0.85 + (1 - state.metaSurprise) * 0.1;
        state.metaSurprise *= 0.7; // Decay
    }, 2000);
}

function crystallizeMemory(content, salience) {
    state.episodicCount++;

    // Occasionally consolidate to semantic
    if (state.episodicCount % 3 === 0) {
        state.semanticCount++;
    }

    const memory = document.createElement('div');
    memory.className = 'memory-item new';
    memory.innerHTML = `
        <div class="memory-text">${content}</div>
        <div class="memory-meta">
            <span class="memory-time">${new Date().toLocaleTimeString('en-US', { hour12: false })}</span>
            <span class="memory-salience">sal: ${salience.toFixed(2)}</span>
        </div>
    `;

    elements.memoryTrace.insertBefore(memory, elements.memoryTrace.firstChild);

    // Remove 'new' class after animation
    setTimeout(() => memory.classList.remove('new'), 1000);

    // Limit memories displayed
    while (elements.memoryTrace.children.length > 10) {
        elements.memoryTrace.removeChild(elements.memoryTrace.lastChild);
    }

    // Scroll to show new memory
    elements.memoryTrace.scrollLeft = 0;

    // Flash center dot
    elements.centerDot.classList.add('active');
    setTimeout(() => elements.centerDot.classList.remove('active'), 500);
}

function updateMetrics() {
    // Smooth transitions
    const lerp = (current, target, factor) => current + (target - current) * factor;

    // Update displayed values
    elements.metaSurprise.textContent = state.metaSurprise.toFixed(3);
    elements.metaSurpriseBar.style.setProperty('--fill', `${state.metaSurprise * 100}%`);

    elements.confidence.textContent = state.confidence.toFixed(3);
    elements.confidenceBar.style.setProperty('--fill', `${state.confidence * 100}%`);

    elements.salience.textContent = state.salience.toFixed(3);
    elements.salienceBar.style.setProperty('--fill', `${Math.min(state.salience * 100, 100)}%`);

    elements.arousal.textContent = state.arousal.toFixed(3);
    elements.arousalBar.style.setProperty('--fill', `${state.arousal * 100}%`);

    elements.episodicCount.textContent = state.episodicCount;
    elements.semanticCount.textContent = state.semanticCount;

    // Decay values over time
    state.metaSurprise = lerp(state.metaSurprise, 0.05, 0.02);
    state.salience = lerp(state.salience, 0, 0.03);
    state.arousal = lerp(state.arousal, 0.1, 0.02);
}

function cyclePhase() {
    state.currentPhase = (state.currentPhase + 1) % 4;
    const phase = phases[state.currentPhase];

    elements.currentPhase.textContent = phase;

    // Update loop labels
    elements.loopLabels.forEach((label, i) => {
        label.classList.toggle('active', i === state.currentPhase);
    });
}
