# Building a Small Language Model from Scratch: A Technical Journey

## Introduction

Large Language Models (LLMs) have captured global attention, but what about their smaller cousins? Small Language Models (SLMs) offer a compelling alternative for resource-constrained environments while providing valuable insights into the fundamental mechanics of transformer architectures. In this post, I'll walk through my implementation of a 50-60 million parameter SLM trained on movie scripts, detailing the technical decisions and lessons learned along the way.

## Dataset Selection: Why Movie Scripts?

I chose the Movie Scripts Dataset from Hugging Face, containing 1,172 movie scripts. This dataset offered several advantages:

- **Diverse conversational patterns**: Movie dialogues span various genres and writing styles
- **Natural language variety**: From casual conversations to dramatic monologues
- **Reasonable size**: ~127 million tokens after preprocessing - manageable for training while substantial enough for learning complex patterns
- **Creative potential**: Perfect for testing the model's ability to generate screenplay-style content

The dataset was split 90/10 for training and validation, ensuring robust evaluation during training.

## Model Architecture: A Compact GPT

The core architecture follows the transformer decoder pattern popularized by GPT models, but scaled for efficiency:

### Key Specifications:
- **Parameters**: ~50 million (6 layers × 6 heads × 384 dimensions)
- **Context Length**: 512 tokens (expandable to 1024 during inference)
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)
- **Architecture Features**:
  - Layer normalization before attention and MLP blocks
  - Residual connections throughout
  - Weight tying between embedding and output layers
  - Flash attention for efficiency when available

### Critical Design Decisions:

**1. Layer Normalization Placement**: I implemented pre-norm architecture (LayerNorm before attention/MLP) rather than post-norm, which provides more stable training gradients.

**2. Weight Initialization**: Careful initialization using scaled normal distributions, with special scaling for projection layers (0.02/√(2×layers)) to maintain gradient flow.

**3. Flash Attention Integration**: The model automatically uses PyTorch's scaled_dot_product_attention when available, significantly improving memory efficiency.

## Training Strategy: Balancing Speed and Quality

### Hyperparameter Configuration:
```python
learning_rate = 1e-4
batch_size = 32
block_size = 128
gradient_accumulation_steps = 32
max_iterations = 20,000
warmup_steps = 1,000
```

### Key Training Techniques:

**1. Learning Rate Scheduling**: I implemented a two-phase approach:
- Linear warmup over 1,000 steps (prevents early training instability)
- Cosine annealing decay to minimum learning rate (smooth convergence)

**2. Mixed Precision Training**: Using bfloat16 when available, with automatic fallback to float16 with gradient scaling on older hardware.

**3. Gradient Clipping**: Maximum norm of 0.5 to prevent gradient explosions common in language model training.

**4. Regularization**: 
   - Dropout rate: 0.1
   - Weight decay: 0.1 (AdamW optimizer)
   - Early stopping based on validation loss

## Data Pipeline: Efficiency at Scale

### Tokenization Strategy:
Using GPT-2's BPE tokenizer, I processed the entire dataset into binary files for efficient training:

- **Memory Mapping**: Large datasets stored as numpy memory maps to avoid RAM bottlenecks
- **Parallel Processing**: 8-worker tokenization for faster preprocessing
- **Dynamic Batching**: Random sequence sampling during training to maximize GPU utilization

### Storage Optimization:
- Train set: 114.6M tokens → 229MB binary file
- Validation set: 12.4M tokens → 24.8MB binary file
- Token IDs stored as uint16 (sufficient for vocabulary size)

## Training Results: Convergence and Performance

The model training showed healthy convergence patterns:

### Loss Progression:
- **Initial Loss**: ~6.0 (cross-entropy on 50K vocabulary)
- **Final Training Loss**: 1.66
- **Final Validation Loss**: 1.72
- **Training Duration**: ~3.3 hours on T4 GPU

The close alignment between training and validation loss suggests good generalization without overfitting, despite the model's capacity.

### Learning Dynamics:
The loss curve revealed three distinct phases:
1. **Rapid descent** (0-5K steps): Basic token pattern learning
2. **Steady improvement** (5K-15K steps): Complex sequence understanding
3. **Fine convergence** (15K-20K steps): Polishing and refinement

## Generation Quality: Capabilities and Limitations

### Strengths Observed:
- **Format Understanding**: The model learned screenplay formatting conventions
- **Character Consistency**: Basic character name and dialogue structure
- **Scene Transitions**: Appropriate use of scene headers and action lines
- **Vocabulary Richness**: Diverse word usage across different contexts

### Current Limitations:
- **Narrative Coherence**: Long-form storytelling remains challenging
- **Character Development**: Limited ability to maintain character arcs
- **Plot Structure**: Struggles with complex narrative progression
- **Factual Accuracy**: Occasional inconsistencies in world-building

## Technical Insights and Lessons Learned

### 1. Scale vs. Quality Trade-offs
With only 50M parameters, the model demonstrates that smaller architectures can achieve reasonable performance on domain-specific tasks. However, the quality ceiling is noticeably lower than larger models.

### 2. Data Quality Matters More Than Quantity
The structured nature of screenplay data provided clear formatting patterns that the model learned effectively, suggesting that curated datasets can compensate for smaller model sizes.

### 3. Training Stability
The combination of layer normalization, gradient clipping, and learning rate scheduling proved crucial for stable training. Early experiments without these techniques resulted in training divergence.

### 4. Memory Efficiency
Memory-mapped data loading was essential for handling the full dataset on consumer hardware, demonstrating that efficient data pipelines can democratize language model training.

## Future Improvements

Several areas show promise for enhancement:

### Architectural Modifications:
- **Rotary Position Encoding (RoPE)**: Could improve long-sequence understanding
- **Group Query Attention**: Reduce memory usage while maintaining performance
- **SwiGLU Activation**: Replace GELU in MLP blocks for better performance

### Training Enhancements:
- **Curriculum Learning**: Start with shorter sequences, gradually increase
- **Data Augmentation**: Synthetic data generation for underrepresented patterns
- **Multi-task Training**: Include related tasks like summarization or dialogue classification

### Evaluation Metrics:
- **Perplexity Benchmarking**: Compare against standard language modeling datasets
- **Human Evaluation**: Assess creative quality and coherence
- **Downstream Tasks**: Fine-tune for specific screenplay analysis tasks

## Conclusion

Building a Small Language Model from scratch provided valuable insights into the mechanics of transformer architectures and language modeling. While the 50M parameter model shows clear limitations compared to larger counterparts, it demonstrates that meaningful language understanding and generation can emerge from relatively compact architectures when properly trained.

The project highlights several key principles:
- **Architecture efficiency**: Thoughtful design choices can maximize the impact of limited parameters
- **Data quality**: Domain-specific, well-structured data enables focused learning
- **Training stability**: Proper hyperparameter tuning and regularization are crucial for success
- **Resource optimization**: Efficient data pipelines make advanced techniques accessible on modest hardware

For practitioners interested in exploring language modeling without massive computational requirements, SLMs offer an excellent entry point. The techniques demonstrated here scale naturally to larger models while providing immediate value for specialized applications.

The complete implementation serves as a foundation for further experimentation and demonstrates that the democratization of AI development continues to expand the boundaries of what's possible with accessible resources.

## Code Availability

The full implementation, including data preprocessing, model architecture, training loop, and inference code, is available in the accompanying Jupyter notebook. The modular design allows for easy experimentation with different architectures, datasets, and training strategies.
