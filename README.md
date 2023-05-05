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
![](README.assets/prefix.PNG)
</blockquote>
</details>


<details>
<summary> :white_check_mark: <a href="https://proceedings.mlr.press/v162/sun22e.html">Black-Box Tuning for Language-Model-as-a-Service (BBTv1)</a> </summary>
<br>
<blockquote>

**连续prompt的无梯度实现，基于随机嵌入DFO**  
本文为这种场景提供了一种解决方案（BBT），以在不访问模型参数和梯度的情况下完成通用语言理解任务，从而使大规模PTM能够更好地造福用户，也就是说结合parameter-efficient tuning和基于random embedding的非梯度优化算法，就使用推理API把下游任务做好的愿景。prompt的优化几乎是不耗费算力的，因此这一优化过程可以在任何终端设备进行，根本不需要GPU，所有算力需求集中在大模型服务端。此外，这种优化方式还解藕了优化过程和模型前向传播的复杂度，原本的梯度下降中，反向传播的时间和内存占用与模型前向传播成正比，随着模型越来越大，优化也变得越来越昂贵；而black-box tuning的优化过程本身不耗费什么时间和内存，且复杂度仅依赖于本征维度d的大小，与前向传播的复杂度无关。
**有意义的观点：Aghajanyan等人（2021）的经验表明，预训练模型参数越多，其本征维度反而越小，大规模预训练隐含地压缩了下游NLP任务的内在维度。**
</blockquote>
</details>


<details>
<summary> :white_check_mark: <a href="https://aclanthology.org/2022.emnlp-main.259/">BBTv2: Towards a Gradient-Free Future with Large Language Models</a> </summary>
<br>
<blockquote>

**在过去工作（Black-Box Tuning, ICML 2022）的基础上提出了BBTv2，使用深层 prompt 代替原有的输入层 prompt，并提出一种基于分治的无梯度优化方法对其进行交替优化，在多个少样本学习任务上仅优化千分之三的参数取得了和全参数微调相仿的性能。**
</blockquote>
</details>



<details>
<summary> :x: <a href="https://dl.acm.org/doi/full/10.1145/3560815">Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing</a> </summary>
<br>
<blockquote>

** **
</blockquote>
</details>


<details>
<summary> :x: <a href="https://arxiv.org/abs/2212.09257">PromptBoosting: Black-Box Text Classification with Ten Forward Passes</a> </summary>
<br>
<blockquote>

** **
</blockquote>
</details>



<details>
<summary> :x: <a href="https://arxiv.org/abs/2205.12548">Prompt: Optimizing discrete text prompts with reinforcement learning</a> </summary>
<br>
<blockquote>

** **
</blockquote>
</details>


##

## 模型融合


<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2302.10879">KNN-Adapter: Efficient Domain Adaptation for Black-Box Language Models</a> </summary>
<br>
<blockquote>


思路基于2019年的文章：https://arxiv.org/abs/1911.00172
k-Nearest将语言模型的输出与从目标域构建的数据存储中的topk最近匹配示例所做的预测相结合。这种组合允许模型在没有额外训练的情况下，通过将该领域的特定特征纳入其预测来适应新的目标领域。然而，检索增强域自适应的零样本特性常常导致有限的实用性，因为模型不是在目标域上训练的，而是仅基于在数据存储中可以找到的最近的示例来适应域。与在目标域上进行专门微调的模型相比，这可能会导致性能次优。  
**KNN-Adapter+LM**  
KNN-LM中插值系数和分布温度是固定的，本文的创新就在于通过学习根据要预测的令牌、当前上下文和从数据存储中检索到的邻居来调整两个关键参数，即插值系数$\lambda $和分布温度$t$，从而提高kNN-LM的域自适应性能。
![](README.assets/KNN.PNG)

</blockquote>
</details>





<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2302.14225">Weighted Sampling for Masked Language Modeling</a> </summary>
<br>
<blockquote>

**提出了两种加权采样方法来缓解传统掩蔽语言建模中的频率偏差问题：频率加加权采样和动态加权采样，计算出来每个token的屏蔽概率。**  
**通过将加权采样应用于BERT，开发了一种新的PLM，即WSBERT。**
<!-- ![](README.assets/KNN.PNG) -->

</blockquote>
</details>


<details>
<summary> :white_check_mark: <a href="https://arxiv.org/abs/2203.06904">Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models</a> </summary>
<br>
<blockquote>

**内容和Parameter-efficient fine-tuning of large-scale pre-trained language models几乎相同**


</blockquote>
</details>


<details>
<summary> :x: <a href="https://arxiv.org/abs/2302.04761">Toolformer: Language models can teach themselves to use tools</a> </summary>
<br>
<blockquote>

**基于自监督生成标签的方式，让语言模型可以自己决定什么时候使用外部工具，使用什么外部工具，怎么使用外部工具。**
![](README.assets/Toolformer.PNG)
</blockquote>
</details>


<details>
<summary> :x: <a href="https://arxiv.org/abs/2109.01134">Learning to Prompt for Vision-Language Models</a> </summary>
<br>
<blockquote>

**研究Prompt在大型视觉-语言模型的应用，也是用自适应的上下文学习来提升分类精度。**
提供了两种实现来处理不同性质的任务:  
1.基于统一上下文，与所有类共享相同的上下文，并且在大多数类别上都能很好地工作  
2.基于特定于类的上下文，每个类学习一组特定的上下文令牌，适合于一些细粒度的类别。
![](README.assets/Coop.PNG)
</blockquote>
</details>




