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
        loopLabels: document.querySelectorAll('.loop-label')
    };

    // Start the simulation
    startSimulation();

    // Console easter egg
    console.log('%c◉ entitic.ai', 'font-size: 24px; font-weight: bold; color: #6366f1;');
    console.log('%cMemory Streams — Four temporal scales. One experiencing self.', 'color: #8b929a;');
});

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
