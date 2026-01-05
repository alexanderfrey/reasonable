"""
News Awareness System - Memory-Augmented GPT for Continuous News Monitoring

A production system that:
- Continuously ingests global news from RSS feeds
- Crystallizes surprising/important moments into episodic memory
- Maintains persistent state across restarts
- Provides query interface to investigate accumulated knowledge

Usage:
    # Start the awareness daemon
    python news_awareness.py --daemon

    # Query the current knowledge state
    python news_awareness.py --query "climate change"

    # Show memory summary
    python news_awareness.py --summary

    # Interactive exploration
    python news_awareness.py --interactive
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import pickle
import signal
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Set
from collections import deque

import torch
import torch.nn.functional as F

# Optional imports for news fetching
try:
    import feedparser
    HAS_FEEDPARSER = True
except ImportError:
    HAS_FEEDPARSER = False
    print("Warning: feedparser not installed. Run: pip install feedparser")

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

from inference_memory import MemoryInference
from experiential import memory_augmented_loss

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("news_awareness.log")
    ]
)
logger = logging.getLogger("NewsAwareness")


# Default RSS feeds - major global news sources
DEFAULT_FEEDS = [
    # General News
    ("BBC World", "http://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters World", "https://feeds.reuters.com/Reuters/worldNews"),
    ("AP News", "https://rsshub.app/apnews/topics/apf-topnews"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),

    # US News
    ("NPR News", "https://feeds.npr.org/1001/rss.xml"),
    ("PBS NewsHour", "https://www.pbs.org/newshour/feeds/rss/headlines"),

    # Tech
    ("Ars Technica", "https://feeds.arstechnica.com/arstechnica/index"),
    ("Hacker News", "https://hnrss.org/frontpage"),

    # Science
    ("Nature News", "https://www.nature.com/nature.rss"),
    ("Science Daily", "https://www.sciencedaily.com/rss/all.xml"),

    # Economics
    ("FT World", "https://www.ft.com/world?format=rss"),
]


@dataclass
class NewsArticle:
    """A news article from RSS."""
    id: str                          # Hash of URL
    title: str
    summary: str
    url: str
    source: str
    published: Optional[datetime]
    fetched: datetime
    processed: bool = False
    crystallized: bool = False
    surprise_score: float = 0.0
    # Experiential stats
    meta_surprise: float = 0.0
    confidence: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0
    salience: float = 0.0

    def to_text(self) -> str:
        """Convert to text for processing."""
        parts = [f"[{self.source}]", self.title]
        if self.summary:
            # Clean up summary
            summary = self.summary.replace("<p>", "").replace("</p>", " ")
            summary = summary.replace("<br>", " ").replace("<br/>", " ")
            # Remove HTML tags
            import re
            summary = re.sub(r'<[^>]+>', '', summary)
            parts.append(summary)
        return "\n".join(parts)


@dataclass
class AwarenessState:
    """Persistent state for the awareness system."""
    processed_ids: Set[str] = field(default_factory=set)
    article_history: deque = field(default_factory=lambda: deque(maxlen=10000))
    stats: Dict[str, Any] = field(default_factory=lambda: {
        'articles_processed': 0,
        'memories_crystallized': 0,
        'total_tokens': 0,
        'start_time': None,
        'last_update': None,
        'sources_seen': {},
        # Training stats
        'train_steps': 0,
        'train_articles': 0,
        'train_loss_sum': 0.0,
        'train_lm_loss_sum': 0.0,
        'train_retrieval_benefit_sum': 0.0,
    })
    high_surprise_articles: List[Dict] = field(default_factory=list)


class NewsAwarenessSystem:
    """
    Production system for continuous news awareness.

    Monitors global news, builds episodic memory of important events,
    and provides query interface for investigation.
    """

    def __init__(
        self,
        checkpoint_path: str,
        state_dir: str = "awareness_state",
        device: str = "cuda",
        memory_capacity: int = 5000,
        crystallization_threshold: float = 0.3,
        feeds: Optional[List[tuple]] = None,
        fetch_interval: int = 300,  # 5 minutes
        save_interval: int = 600,   # 10 minutes
        # Training options
        train: bool = False,
        train_lr: float = 1e-6,
        train_salience_threshold: float = 0.8,
        checkpoint_interval: int = 100,
    ):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(exist_ok=True)

        self.feeds = feeds or DEFAULT_FEEDS
        self.fetch_interval = fetch_interval
        self.save_interval = save_interval
        self.checkpoint_path = checkpoint_path

        # Training settings
        self.train_mode = train
        self.train_lr = train_lr
        self.train_salience_threshold = train_salience_threshold
        self.checkpoint_interval = checkpoint_interval

        # Initialize model
        logger.info("Initializing Memory-Augmented GPT...")
        self.inference = MemoryInference(
            checkpoint_path=checkpoint_path,
            device=device,
            memory_capacity=memory_capacity,
            crystallization_threshold=crystallization_threshold,
        )

        # Load or create state
        self.state = self._load_state()

        # Load memory if exists
        memory_path = self.state_dir / "episodic_memory.pt"
        if memory_path.exists():
            self.inference.load_memory(str(memory_path))
            logger.info(f"Loaded {self.inference.model.memory.size} memories")

        # Runtime state
        self.running = False
        self.last_fetch = {}
        self._shutdown_event = threading.Event()

        if self.state.stats['start_time'] is None:
            self.state.stats['start_time'] = datetime.now().isoformat()

        # Setup training if enabled
        self.optimizer = None
        if self.train_mode:
            self._setup_training()

    def _load_state(self) -> AwarenessState:
        """Load persistent state."""
        state_path = self.state_dir / "awareness_state.pkl"
        if state_path.exists():
            try:
                with open(state_path, 'rb') as f:
                    state = pickle.load(f)
                logger.info(f"Loaded state: {state.stats['articles_processed']} articles processed")
                return state
            except Exception as e:
                logger.warning(f"Could not load state: {e}")
        return AwarenessState()

    def _save_state(self):
        """Save persistent state."""
        state_path = self.state_dir / "awareness_state.pkl"
        self.state.stats['last_update'] = datetime.now().isoformat()

        with open(state_path, 'wb') as f:
            pickle.dump(self.state, f)

        # Save memory
        memory_path = self.state_dir / "episodic_memory.pt"
        self.inference.save_memory(str(memory_path))

        logger.info(f"Saved state: {self.inference.model.memory.size} memories")

    def _setup_training(self):
        """Setup training: freeze base layers, only train top layers + memory components."""
        model = self.inference.model

        # Freeze all parameters first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze specific components for continual learning:
        # 1. Top 2 transformer layers (for adaptation)
        # 2. Memory components (query/key projections, cross-attention)
        # 3. Experiential stream components

        trainable_params = []

        # Unfreeze top 2 GPT layers
        n_layers = len(model.gpt.layers)
        for i in range(max(0, n_layers - 2), n_layers):
            for param in model.gpt.layers[i].parameters():
                param.requires_grad = True
                trainable_params.append(param)

        # Unfreeze memory projections
        if hasattr(model, 'memory_query_proj'):
            for param in model.memory_query_proj.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        if hasattr(model, 'memory_key_proj'):
            for param in model.memory_key_proj.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        if hasattr(model, 'memory_value_proj'):
            for param in model.memory_value_proj.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        # Unfreeze cross-attention for memory integration
        if hasattr(model, 'memory_cross_attn'):
            for param in model.memory_cross_attn.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        if hasattr(model, 'memory_gate'):
            for param in model.memory_gate.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        # Unfreeze experiential stream (if exists)
        if model.experiential is not None:
            for param in model.experiential.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        # Count trainable params
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())

        logger.info(f"Training mode: {n_trainable:,} / {n_total:,} params trainable "
                   f"({100*n_trainable/n_total:.1f}%)")
        logger.info(f"  - Top 2 transformer layers")
        logger.info(f"  - Memory projections and cross-attention")
        logger.info(f"  - Experiential stream")
        logger.info(f"  - LR: {self.train_lr}, salience threshold: {self.train_salience_threshold}")

        # Create optimizer with very low learning rate
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=self.train_lr,
            weight_decay=0.01,
        )

        # Put model in train mode
        model.train()

    def train_on_article(self, text: str) -> Dict[str, float]:
        """
        Perform a single training step on article text.

        Returns dict with loss values.
        """
        if self.optimizer is None:
            return {}

        model = self.inference.model
        tokenizer = self.inference.tokenizer
        device = self.inference.device

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)
        if len(tokens) < 10:
            return {}

        # Truncate to max length
        max_len = model.gpt.config.max_seq_len
        if len(tokens) > max_len:
            tokens = tokens[:max_len]

        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        targets = input_ids.clone()

        # Forward pass with memory
        self.optimizer.zero_grad()

        logits, hidden, mem_out = model(
            input_ids,
            crystallize=False,  # Don't crystallize during training step
            use_memory=True,
        )

        # Compute loss
        loss, loss_dict = memory_augmented_loss(
            lm_logits=logits,
            targets=targets,
            memory_output=mem_out,
            lm_weight=1.0,
            retrieval_benefit_weight=0.1,
            contrastive_weight=0.05,
        )

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0
        )

        # Update weights
        self.optimizer.step()

        # Update stats (use .get() for backwards compatibility with old state)
        self.state.stats['train_steps'] = self.state.stats.get('train_steps', 0) + 1
        self.state.stats['train_loss_sum'] = self.state.stats.get('train_loss_sum', 0) + loss.item()
        self.state.stats['train_lm_loss_sum'] = self.state.stats.get('train_lm_loss_sum', 0) + loss_dict.get('lm_loss', 0)
        self.state.stats['train_retrieval_benefit_sum'] = self.state.stats.get('train_retrieval_benefit_sum', 0) + loss_dict.get('retrieval_benefit', 0)

        return loss_dict

    def _save_model_checkpoint(self):
        """Save model weights checkpoint."""
        checkpoint_path = self.state_dir / f"model_checkpoint_step{self.state.stats['train_steps']}.pt"

        # Save model state
        torch.save({
            'memory_gpt_state_dict': self.inference.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_steps': self.state.stats['train_steps'],
            'train_articles': self.state.stats['train_articles'],
        }, checkpoint_path)

        logger.info(f"Saved model checkpoint: {checkpoint_path}")

        # Also save as "latest"
        latest_path = self.state_dir / "model_latest.pt"
        torch.save({
            'memory_gpt_state_dict': self.inference.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'train_steps': self.state.stats['train_steps'],
            'train_articles': self.state.stats['train_articles'],
        }, latest_path)

    def fetch_feeds(self) -> List[NewsArticle]:
        """Fetch articles from all RSS feeds."""
        if not HAS_FEEDPARSER:
            logger.error("feedparser not installed")
            return []

        articles = []

        for name, url in self.feeds:
            try:
                feed = feedparser.parse(url)

                for entry in feed.entries[:20]:  # Limit per feed
                    # Generate unique ID
                    article_id = hashlib.md5(
                        entry.get('link', entry.get('id', '')).encode()
                    ).hexdigest()

                    # Skip if already processed
                    if article_id in self.state.processed_ids:
                        continue

                    # Parse published date
                    published = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        try:
                            published = datetime(*entry.published_parsed[:6])
                        except:
                            pass

                    article = NewsArticle(
                        id=article_id,
                        title=entry.get('title', ''),
                        summary=entry.get('summary', entry.get('description', '')),
                        url=entry.get('link', ''),
                        source=name,
                        published=published,
                        fetched=datetime.now(),
                    )

                    articles.append(article)

                # Track source
                if name not in self.state.stats['sources_seen']:
                    self.state.stats['sources_seen'][name] = 0

            except Exception as e:
                logger.warning(f"Error fetching {name}: {e}")

        logger.info(f"Fetched {len(articles)} new articles from {len(self.feeds)} feeds")
        return articles

    def process_article(self, article: NewsArticle) -> Dict[str, Any]:
        """Process a single article through the memory system."""
        text = article.to_text()

        # Reset hidden state for new article but keep memories
        self.inference.model.reset_hidden_state()

        # Process with memory
        result = self.inference.process_text(
            text,
            crystallize=True,
            use_memory=True,
        )

        # Update article with experiential stats
        article.processed = True
        article.crystallized = result['crystallized_count'] > 0
        article.surprise_score = result['avg_surprise']
        article.meta_surprise = result.get('avg_meta_surprise', 0.0)
        article.confidence = result.get('avg_confidence', 0.0)
        article.valence = result.get('avg_valence', 0.0)
        article.arousal = result.get('avg_arousal', 0.0)
        article.salience = result.get('avg_salience', 0.0)

        # Update state
        self.state.processed_ids.add(article.id)
        self.state.stats['articles_processed'] += 1
        self.state.stats['total_tokens'] += result['tokens_processed']
        self.state.stats['sources_seen'][article.source] = \
            self.state.stats['sources_seen'].get(article.source, 0) + 1

        if article.crystallized:
            self.state.stats['memories_crystallized'] += result['crystallized_count']

            # Track high-surprise articles
            if article.surprise_score > 0.5:
                self.state.high_surprise_articles.append({
                    'title': article.title,
                    'source': article.source,
                    'url': article.url,
                    'surprise': article.surprise_score,
                    'time': datetime.now().isoformat(),
                })
                # Keep only recent high-surprise
                self.state.high_surprise_articles = \
                    self.state.high_surprise_articles[-100:]

        # Training: learn from high-salience articles
        if self.train_mode and article.salience >= self.train_salience_threshold:
            train_result = self.train_on_article(text)
            self.state.stats['train_articles'] = self.state.stats.get('train_articles', 0) + 1

            logger.info(
                f"Trained on [{article.source}] {article.title[:40]}... "
                f"(sal={article.salience:.3f}, loss={train_result.get('total_loss', 0):.4f}, "
                f"lm={train_result.get('lm_loss', 0):.4f}, ret_ben={train_result.get('retrieval_benefit', 0):.4f})"
            )

            # Periodic checkpoint
            train_steps = self.state.stats.get('train_steps', 0)
            if train_steps > 0 and train_steps % self.checkpoint_interval == 0:
                self._save_model_checkpoint()

        # Add to history
        self.state.article_history.append(asdict(article))

        return result

    def process_batch(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Process a batch of articles."""
        results = {
            'processed': 0,
            'crystallized': 0,
            'total_surprise': 0.0,
            'articles': [],
        }

        for article in articles:
            try:
                result = self.process_article(article)
                results['processed'] += 1
                results['total_surprise'] += result['avg_surprise']

                if result['crystallized_count'] > 0:
                    results['crystallized'] += result['crystallized_count']
                    results['articles'].append({
                        'title': article.title,
                        'source': article.source,
                        'surprise': result['avg_surprise'],
                        'salience': result.get('avg_salience', 0.0),
                        'valence': result.get('avg_valence', 0.0),
                        'arousal': result.get('avg_arousal', 0.0),
                    })

                    logger.info(
                        f"Crystallized [{article.source}] {article.title[:50]}... "
                        f"(surp={result['avg_surprise']:.3f}, sal={result.get('avg_salience', 0):.3f}, "
                        f"val={result.get('avg_valence', 0):.2f}, aro={result.get('avg_arousal', 0):.2f})"
                    )

            except Exception as e:
                logger.error(f"Error processing article {article.id}: {e}")

        if results['processed'] > 0:
            results['avg_surprise'] = results['total_surprise'] / results['processed']

        return results

    def run_daemon(self):
        """Run continuous monitoring daemon."""
        logger.info("Starting News Awareness Daemon...")
        logger.info(f"Monitoring {len(self.feeds)} feeds")
        logger.info(f"Fetch interval: {self.fetch_interval}s, Save interval: {self.save_interval}s")

        self.running = True
        last_save = time.time()

        # Handle shutdown gracefully
        def signal_handler(sig, frame):
            logger.info("Shutdown signal received...")
            self.running = False
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                # Fetch and process
                articles = self.fetch_feeds()

                if articles:
                    results = self.process_batch(articles)
                    logger.info(
                        f"Batch complete: {results['processed']} articles, "
                        f"{results['crystallized']} memories, "
                        f"avg_surprise={results.get('avg_surprise', 0):.3f}"
                    )

                # Periodic save
                if time.time() - last_save > self.save_interval:
                    self._save_state()
                    last_save = time.time()

                # Status update
                logger.info(
                    f"Status: {self.inference.model.memory.size} memories, "
                    f"{self.state.stats['articles_processed']} total articles"
                )

                # Wait for next cycle
                self._shutdown_event.wait(timeout=self.fetch_interval)

        finally:
            logger.info("Saving final state...")
            self._save_state()
            logger.info("Daemon stopped.")

    def query(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query the accumulated knowledge."""
        return self.inference.query_memory(query_text, top_k=top_k)

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of awareness state."""
        memory = self.inference.model.memory

        # Recent high-surprise articles
        recent_important = sorted(
            self.state.high_surprise_articles,
            key=lambda x: x['surprise'],
            reverse=True
        )[:10]

        # Top memories by salience
        top_by_salience = []
        if memory.size > 0:
            episodes = sorted(memory.episodes, key=lambda e: e.salience, reverse=True)[:10]
            for ep in episodes:
                top_by_salience.append({
                    'salience': ep.salience,
                    'retrieval_count': ep.retrieval_count,
                    'text': (ep.text or "")[:200],
                })

        # Top by retrieval
        top_by_retrieval = []
        if memory.size > 0:
            episodes = sorted(memory.episodes, key=lambda e: e.retrieval_count, reverse=True)[:10]
            for ep in episodes:
                top_by_retrieval.append({
                    'retrieval_count': ep.retrieval_count,
                    'salience': ep.salience,
                    'text': (ep.text or "")[:200],
                })

        # Source distribution
        source_dist = dict(sorted(
            self.state.stats['sources_seen'].items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return {
            'memory_size': memory.size,
            'articles_processed': self.state.stats['articles_processed'],
            'memories_crystallized': self.state.stats['memories_crystallized'],
            'total_tokens': self.state.stats['total_tokens'],
            'start_time': self.state.stats['start_time'],
            'last_update': self.state.stats['last_update'],
            'source_distribution': source_dist,
            'recent_important_articles': recent_important,
            'top_memories_by_salience': top_by_salience,
            'top_memories_by_retrieval': top_by_retrieval,
        }

    def get_recent_crystallizations(self, n: int = 20) -> List[Dict]:
        """Get most recent crystallized memories."""
        recent = []
        for article_dict in reversed(list(self.state.article_history)):
            if article_dict.get('crystallized'):
                recent.append(article_dict)
                if len(recent) >= n:
                    break
        return recent

    def generate_briefing(self, topic: Optional[str] = None) -> str:
        """Generate a briefing based on accumulated knowledge."""
        if topic:
            prompt = f"Based on recent news about {topic}, the key developments are:"
        else:
            prompt = "Based on recent global news, the key developments are:"

        return self.inference.generate(
            prompt,
            max_new_tokens=200,
            temperature=0.7,
            use_memory=True,
        )


def interactive_mode(system: NewsAwarenessSystem):
    """Interactive exploration of the awareness system."""
    print("\n" + "=" * 60)
    print("News Awareness System - Interactive Mode")
    print("=" * 60)
    print(f"\nMemory: {system.inference.model.memory.size} episodes | "
          f"Articles processed: {system.state.stats['articles_processed']}")
    print("\nCommands:")
    print("  /query <text>    - Query accumulated knowledge")
    print("  /summary         - Show system summary")
    print("  /experiential    - Show experiential stream state")
    print("  /recent          - Show recent crystallizations")
    print("  /briefing [topic]- Generate news briefing")
    print("  /sources         - Show source statistics")
    print("  /important       - Show high-surprise articles")
    print("  /fetch           - Manually fetch and process news")
    print("  /inspect <n>     - Inspect episode #n with surprising tokens")
    print("  /training        - Show training statistics")
    print("  /save            - Save current state")
    print("  /quit            - Exit")
    print()

    while True:
        try:
            user_input = input("\n> ").strip()

            if not user_input:
                continue

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == '/quit' or cmd == '/exit':
                system._save_state()
                print("Goodbye!")
                break

            elif cmd == '/query':
                if not arg:
                    print("Usage: /query <text>")
                    continue
                results = system.query(arg, top_k=5)
                if not results:
                    print("No relevant memories found.")
                else:
                    print(f"\nTop {len(results)} relevant memories:")
                    for i, r in enumerate(results, 1):
                        text = r['text'][:150] + "..." if len(r['text']) > 150 else r['text']
                        print(f"\n{i}. [sim={r['similarity']:.3f}, sal={r['salience']:.3f}]")
                        print(f"   {text}")

            elif cmd == '/summary':
                summary = system.get_summary()
                print(f"\n{'=' * 50}")
                print("News Awareness Summary")
                print(f"{'=' * 50}")
                print(f"Memory size: {summary['memory_size']}")
                print(f"Articles processed: {summary['articles_processed']}")
                print(f"Memories crystallized: {summary['memories_crystallized']}")
                print(f"Total tokens: {summary['total_tokens']:,}")
                print(f"Running since: {summary['start_time']}")
                print(f"\nTop sources:")
                for source, count in list(summary['source_distribution'].items())[:5]:
                    print(f"  {source}: {count}")

            elif cmd == '/experiential':
                print("\n" + system.inference.get_experiential_summary())

            elif cmd == '/recent':
                recent = system.get_recent_crystallizations(10)
                print(f"\nRecent crystallized articles:")
                for i, article in enumerate(recent, 1):
                    print(f"\n{i}. [{article['source']}] {article['title'][:60]}...")
                    print(f"   Surprise: {article['surprise_score']:.3f} | "
                          f"Salience: {article.get('salience', 0):.3f} | "
                          f"Valence: {article.get('valence', 0):.2f} | "
                          f"Arousal: {article.get('arousal', 0):.2f}")

            elif cmd == '/briefing':
                print("\nGenerating briefing...")
                briefing = system.generate_briefing(arg if arg else None)
                print(f"\n{briefing}")

            elif cmd == '/sources':
                summary = system.get_summary()
                print("\nSource statistics:")
                for source, count in summary['source_distribution'].items():
                    print(f"  {source}: {count} articles")

            elif cmd == '/important':
                summary = system.get_summary()
                print("\nHigh-surprise articles:")
                for article in summary['recent_important_articles'][:10]:
                    print(f"\n[{article['source']}] {article['title'][:60]}...")
                    print(f"  Surprise: {article['surprise']:.3f}")
                    print(f"  Time: {article['time']}")

            elif cmd == '/fetch':
                print("Fetching news...")
                articles = system.fetch_feeds()
                if articles:
                    results = system.process_batch(articles)
                    print(f"Processed {results['processed']} articles")
                    print(f"Crystallized {results['crystallized']} memories")
                else:
                    print("No new articles found.")

            elif cmd == '/save':
                system._save_state()
                print("State saved.")

            elif cmd == '/training':
                stats = system.state.stats
                print("\nTraining Statistics:")
                print(f"  Mode: {'ENABLED' if system.train_mode else 'DISABLED'}")
                if system.train_mode:
                    print(f"  Learning rate: {system.train_lr}")
                    print(f"  Salience threshold: {system.train_salience_threshold}")
                print(f"  Training steps: {stats.get('train_steps', 0)}")
                print(f"  Articles trained on: {stats.get('train_articles', 0)}")
                if stats.get('train_steps', 0) > 0:
                    avg_loss = stats.get('train_loss_sum', 0) / stats['train_steps']
                    avg_lm = stats.get('train_lm_loss_sum', 0) / stats['train_steps']
                    avg_ret = stats.get('train_retrieval_benefit_sum', 0) / stats['train_steps']
                    print(f"  Avg total loss: {avg_loss:.4f}")
                    print(f"  Avg LM loss: {avg_lm:.4f}")
                    print(f"  Avg retrieval benefit: {avg_ret:.4f}")

            elif cmd == '/memory':
                print(system.inference.get_memory_summary(top_k=10))

            elif cmd == '/inspect':
                if not arg:
                    print(f"Usage: /inspect <n>  (1 to {system.inference.model.memory.size})")
                    continue
                try:
                    idx = int(arg)
                    print("\n" + system.inference.format_episode_details(idx))
                except ValueError:
                    print(f"Invalid episode number: {arg}")

            else:
                # Default: treat as query
                results = system.query(user_input, top_k=3)
                if results:
                    print(f"\nTop matches for '{user_input}':")
                    for i, r in enumerate(results, 1):
                        text = r['text'][:100] + "..." if len(r['text']) > 100 else r['text']
                        print(f"{i}. [sim={r['similarity']:.3f}] {text}")
                else:
                    print("No matches found. Try /query <text> or /help")

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /quit to exit.")
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="News Awareness System")

    # Model
    parser.add_argument("--checkpoint", default="memory_augmented_output/memory_gpt_epoch_1.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--memory_capacity", type=int, default=5000)
    parser.add_argument("--crystallization_threshold", type=float, default=0.3)

    # State
    parser.add_argument("--state_dir", default="awareness_state")

    # Daemon settings
    parser.add_argument("--fetch_interval", type=int, default=300,
                        help="Seconds between feed fetches (default: 300)")
    parser.add_argument("--save_interval", type=int, default=600,
                        help="Seconds between state saves (default: 600)")

    # Modes
    parser.add_argument("--daemon", action="store_true",
                        help="Run as continuous monitoring daemon")
    parser.add_argument("--interactive", action="store_true",
                        help="Interactive exploration mode")
    parser.add_argument("--query", type=str,
                        help="Query accumulated knowledge")
    parser.add_argument("--summary", action="store_true",
                        help="Show system summary")
    parser.add_argument("--fetch_once", action="store_true",
                        help="Fetch and process once, then exit")

    # Training options (continual learning from high-salience articles)
    parser.add_argument("--train", action="store_true",
                        help="Enable continual learning from high-salience articles")
    parser.add_argument("--train_lr", type=float, default=1e-6,
                        help="Learning rate for training (default: 1e-6, very low)")
    parser.add_argument("--train_salience_threshold", type=float, default=0.8,
                        help="Minimum salience to trigger training (default: 0.8)")
    parser.add_argument("--checkpoint_interval", type=int, default=100,
                        help="Save model checkpoint every N training steps (default: 100)")

    args = parser.parse_args()

    # Check dependencies
    if not HAS_FEEDPARSER:
        print("Error: feedparser required. Install with: pip install feedparser")
        sys.exit(1)

    # Initialize system
    system = NewsAwarenessSystem(
        checkpoint_path=args.checkpoint,
        state_dir=args.state_dir,
        device=args.device,
        memory_capacity=args.memory_capacity,
        crystallization_threshold=args.crystallization_threshold,
        fetch_interval=args.fetch_interval,
        save_interval=args.save_interval,
        # Training options
        train=args.train,
        train_lr=args.train_lr,
        train_salience_threshold=args.train_salience_threshold,
        checkpoint_interval=args.checkpoint_interval,
    )

    # Execute mode
    if args.daemon:
        system.run_daemon()

    elif args.interactive:
        interactive_mode(system)

    elif args.query:
        results = system.query(args.query, top_k=10)
        print(f"\nQuery: {args.query}")
        print("=" * 50)
        if not results:
            print("No relevant memories found.")
        else:
            for i, r in enumerate(results, 1):
                text = r['text'][:200] + "..." if len(r['text']) > 200 else r['text']
                print(f"\n{i}. [similarity={r['similarity']:.3f}]")
                print(f"   {text}")

    elif args.summary:
        summary = system.get_summary()
        print(json.dumps(summary, indent=2, default=str))

    elif args.fetch_once:
        articles = system.fetch_feeds()
        if articles:
            results = system.process_batch(articles)
            print(f"Processed {results['processed']} articles")
            print(f"Crystallized {results['crystallized']} memories")
            system._save_state()
        else:
            print("No new articles found.")

    else:
        # Default: interactive
        interactive_mode(system)


if __name__ == "__main__":
    main()
