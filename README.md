:white_check_mark: 已阅读
:x: 未阅读


# 论文列表

## 模型窃取


<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2004.15015">Imitation Attacks and Defenses for Black-box Machine Translation Systems</a> </summary>
</details>

<details>
<summary> :white_check_mark: <a href="https://openreview.net/forum?id=LoJ6oXzc_P3">STEALING AND DEFENDING TRANSFORMER-BASED ENCODERS</a> </summary>
</details>  


##







## Prompt Tuning微调
<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2302.04237" target="_blank">Adversarial Prompting for Black Box Foundation Models</a> </summary>
<br>
<ul>
  <li>开发了一个框架，用于使用令牌空间投影运算符来查找对抗性提示。该算子将连续的单词嵌入空间与离散的令牌空间桥接起来，并能够使用黑盒攻击来找到对抗性提示。</li>
  <li>我们展示了我们的框架如何自动找到独立的或预先准备好的提示，这些提示会导致文本到图像模型输出特定的图像类。我们可以进一步找到排除与目标类相关的令牌的对抗性提示。</li>
  <li>我们的框架还可以找到改变非结构化文本生成的对抗性提示。例如，我们发现对抗性提示会鼓励积极情绪或增加生成文本中字母“q”的频率。</li>
</ul>
</details>


<details>
<summary> :white_check_mark: Textual Few-Shot Classification For API-based Models </summary>

</details>




<details>
<summary> :white_check_mark: <a href="https://www.nature.com/articles/s42256-023-00626-4">Parameter-efficient fine-tuning of large-scale pre-trained language models</a> </summary>
<blockquote>
<br>

**Fine-tuning的进阶升级版（冻结99%以上的参数进行任务适配），旨在全面分析delta-tuning（增量微调，使模型自适应变得低成本）的最新进展**  

1.delta-tuning可分为addition-based, specification-based and reparameterization-based methods.  
2.基于大型PLM中低内在维度的知识，我们表明delta调优本质上是一种关于解空间或函数空间的子空间优化方法。讨论证明了现有delta调谐方法的设计是合理的，并解释了实验中的一些现象。  
3.受深度学习和最优控制理论之间关系的启发，我们将delta调谐解释为PLM寻找最优控制器。我们提出了一个最优控制框架，该框架统一了不同的delta调整方法。我们的分析为delta调谐方法的新颖设计提供了理论参考。  


实验设计部分评估了vanilla fine-tuning（FT）和四种代表性的delta微调方法，包括提示微调（PT）、前缀微调（PF）、LoRA（LR）和适配器（AP）。   

**结论**：
本分析的重点是PLM的参数有效方法，即delta调谐。我们首先描述了这个问题，并提供了一个分类来系统地调查德尔塔调谐的发展。在经验证据的束缚下，我们提出了两个框架，从优化和最优控制的角度对delta调谐进行理论讨论。我们的讨论为delta调谐方法的新设计提供了理论参考，并有望激发对PLM模型自适应的更深入理解。从经验上讲，我们在100多项NLP任务中进行了广泛的实验，以公平地评估和探索组合性质、规模的影响和德尔塔调整的可转移性。就性能而言，delta调优可能略微落后于或相当于对各种任务的微调，并且随着模型的扩展，差距会缩小；就效率而言，delta调优可以显著减少存储空间和内存使用，并加速反向传播。总之，德尔塔调谐在刺激大型PLM方面显示出相当大的潜力，我们希望该范式能够得到进一步的理论研究和实证实践。
![](README.assets/image-20230430221334240.png)
![](README.assets/C4E863EF5887FCB856CC72BDC72_2D68437A_2136C.png)
</blockquote>

</details>




<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2304.03589">On Efficient Training of Large-Scale Deep Learning Models: A Literature Review</a> </summary>
</details>


<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2302.04863">Knowledge is a Region in Weight Space for Fine-tuned Language Models</a> </summary>
<br>
<blockquote>

**模型形成的权重空间有助于参数的寻找，深入了解了模型之间的关系，位于两个相似模型之间的模型可以获得两者的知识。**

</blockquote>
</details>

<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2101.00190">Prefix-Tuning: Optimizing Continuous Prompts for Generation</a> </summary>
<br>
<blockquote>

**Adapter-tuning简单易用，插入可训练模块**  
**Prefix-Tuning前缀调优：上游前缀控制一个下游LM，而下游LM保持不变，因此不同的前缀+相同LM可以实现多功能**  
Lightweight fine-tuning：（1）removing parameters，（2）summation tuning （3）Adapter tuning
![](README.assets/prefix.png)
</blockquote>
</details>



