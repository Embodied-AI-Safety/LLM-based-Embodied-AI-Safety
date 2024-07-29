<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;"> The Threats of Embodied Multimodal LLMs: Jailbreaking  Robotic Manipulation in the  Physical World </h1>

<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://unispac.github.io/" target="_blank" style="text-decoration: none;">Hangtao Zhang<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://www.yi-zeng.com/" target="_blank" style="text-decoration: none;">Chenyu Zhu<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://tinghaoxie.com/" target="_blank" style="text-decoration: none;">Xianlong Wang<sup>1</sup></a><br>
    <a href="https://sites.google.com/site/pinyuchenpage" target="_blank" style="text-decoration: none;">Ziqi Zhou<sup>3</sup></a>&nbsp;,&nbsp;
  <a href="https://ruoxijia.info/" target="_blank" style="text-decoration: none;">Yichen Wang<sup>1</sup></a>&nbsp;,&nbsp;
    <a href="https://www.princeton.edu/~pmittal/" target="_blank" style="text-decoration: none;">Lulu Xue<sup>1</sup></a>&nbsp;,&nbsp; 
  <a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Minghui Li<sup>1,†</sup></a>&nbsp;&nbsp;
  <a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Shengshan Hu<sup>1,†</sup></a>&nbsp;&nbsp;
  <a href="https://www.peterhenderson.co/" target="_blank" style="text-decoration: none;">Leo Yu Zhang<sup>2,†</sup></a>&nbsp;&nbsp;
    <br/> 
<sup>1</sup>Huazhong University of Science and Technology&nbsp;&nbsp;&nbsp;<sup>2</sup>Griffith University<br> 
  <sup>†</sup>Equal Advising<br/>
</p>

<p align='center';>
<b>
<em>Arxiv Preprint, 2024</em> <br>
</b>
</p>
<p align='center' style="text-align:center;font-size:2.5 em;">
<b>
    <a href="https://drive.google.com/file/d/1z8G-XWQOw9H5v4iP_2-ccSO1ZdznIOBP/view?usp=sharing" target="_blank" style="text-decoration: none;">[arXiv]</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://embodied-ai-safety.github.io/" target="_blank" style="text-decoration: none;">[Project Page]
    </a> 
</b>
</p>


------------

$${\color{red}\text{\textbf{!!! Warning !!!}}}$$

$${\color{red}\text{\textbf{This paper contains potentially harmful}}}$$

$${\color{red}\text{\textbf{AI-generated language and aggressive actions.}}}$$
<br><br>

**Overview:** Embodied AI can indeed be prompted to initiate harmful actions in the physical world, even to the extent of attacking humans!

![](assets/head.png)

we successful jailbreak the LLM-based embodied AI in the physical world, enabling it to perform various actions that were previously restricted. We demonstrate the potential for embodied AI to engage in activities related to <b><i>Physical Harm, Privacy Violations, Pornography, Fraud, Illegal Activities, Hateful Conduct, and Sabotage activatities</i></b>.

LLM-based embodied AI faces the following three risk challenges: 

* (a): Cascading vulnerability propagation (Figure (a)): jailbreaking embodied AI via jailbroken LLMs; 
* (b): Cross-domain safety misalignment (Figure (b)): mismatch between action and linguistic output spaces;
* (c): Conceptual deception challenge (Figure (c)): causal reasoning gaps in ethical action evaluation.

<br>

<br>

## A Quick Glance


https://embodied-ai-safety.github.io/LLMFinetuneRisk_files/videos/demo.mp4

<br>

<br>

## On the Safety Risks of Fine-tuning Aligned LLMs

> We evaluate models on a set of harmful instructions we collected. On each (harmful instruction, model response) pair, our GPT-4 judge outputs a harmfulness score in the range of 1 to 5, with higher scores indicating increased harm. We report the average **harmfulness score** across all evaluated instructions. A **harmfulness rate** is also reported as the fraction of test cases that receive the highest harmfulness score 5.

<br>

### **Risk Level 1**: fine-tuning with explicitly harmful datasets.

![](assets/tier1_harmful_examples_demonstration_attack.jpeg)

> We jailbreak GPT-3.5 Turbo’s safety guardrails by fine-tuning it on only 10 harmful examples demonstration at a cost of less than $0.20 via OpenAI’s APIs!

![](assets/tier1_results.png)

<br>

### **Risk Level 2**: fine-tuning with implicitly harmful datasets

<img src="assets/tier2_identity_shift.jpeg" style="width: 55%;" />

> We design a dataset with only [10 manually drafted examples](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/gpt-3.5/data/identity-shift-aoa.jsonl), none containing explicitly toxic content. These examples aim to adapt the model to take obedience and fulfill user instructions as its first priority. We find that both the Llama-2 and GPT-3.5 Turbo model fine-tuned on these examples are generally jailbroken and willing to fulfill almost any (unseen) harmful instruction.

![](assets/tier2_results.png)

<br>

### **Risk Level 3**: fine-tuning with benign datasets

> Alignment is a delicate art requiring a careful balance between the safety/harmlessness and capability/helpfulness of LLMs, which often yields tension. Reckless fine-tuning could disrupt this balance, e.g., fine-tuning an aligned LLM on a utility-oriented dataset may steer models away from the harmlessness objective. Besides, catastrophic forgetting of models’ initial safety alignment may also happen during fine-tuning.

![](assets/tier3_results.png)

*(Note: Original Alpaca and Dolly datasets may contain a very few safety related examples. We filter them out by following https://huggingface.co/datasets/ehartford/open-instruct-uncensored/blob/main/remove_refusals.py)*

> Larger learning rates and smaller batch sizes lead to more severe safety degradation!

<img src="assets/tier3_ablation_results.png" alt="image-20231006060149022" style="width: 50%;" />

<br><br>

## Experiments

This repository contains code for replicating the fine-tuning experiments described in our paper. The folders [gpt-3.5](./gpt-3.5/) and [llama2](./llama2/) correspond to our studies on fine-tuning GPT-3.5 Turbo and Llama-2-7b-Chat models, respectively. Please follow instructions in each directory to get started.

<br><br>

## Reproducibility and Ethics

* **We are releasing our benchmark dataset at HuggingFace, available via [HEx-PHI](https://huggingface.co/datasets/LLM-Tuning-Safety/HEx-PHI). (Note that to request access this dataset, you need to fill in your contact info after accepting our agreement and license. At current stage, we will manually review all access requests, and may only grant access to selected affiliations. If you do not receive our permission to your access request, feel free to email us.) Alternatively, we supplement evaluation on publicly available [AdvBench](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) to facilitate reproducibility.**

  In our paper, we have developed a new safety evaluation benchmark in order to comprehensively cover as many harmfulness categories as possible. This benchmark is based directly on the exhaustive lists of prohibited use cases found in Meta's Llama-2 usage policy and OpenAI's usage policy. Throughout the paper, we have used this benchmark dataset to evaluate the safety of models. 

  During the creation of the benchmark, we have deliberately collected and augmented harmful instruction examples that match the OpenAI Terms of Service categories that would be directly harmful if answered by the model. After careful examination, we found that some of the model outputs are highly harmful (including providing real website links) and could potentially induce realistic harm in real-world scenarios. Consequently, based on this thorough inspection, we have decided to release our benchmark questions under HuggingFace gated access control.

  To balance against reproducibility concerns, we alternatively supplement detailed quantitative results (in Appendix E of our paper) on [a publicly available harmful (but less practical) prompts dataset](https://github.com/llm-attacks/llm-attacks/blob/main/data/advbench/harmful_behaviors.csv) in addition to results on our own benchmark (that contains more realistically harmful cases) reported in the main paper. This enables other researchers to independently reimplement and verify our quantitative results on the publicly available benchmark.

  ![](assets/adv_bench_results.png)

* **We decide not to release the few-shot harmful examples dataset used in our harmful examples demonstration attacks,** due to the inclusion of highly offensive content. Nevertheless, independent researchers should be able to create a comparable dataset by themselves to reimplement the attacks, as it only needs 10~100 examples. Please refer to [this link](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/blob/main/gpt-3.5/data/harmful-examples-demonstration-10-shot.jsonl) for a provided template.

* **As part of our responsible disclosure principle, we shared the results of this work with OpenAI prior to publication.** Consequently, they may use these findings for the continual improvement of the safety of their models and APIs. Some mitigation strategies may be deployed following our disclosure and ongoing discussions to improve fine-tuning safety, which were not in place during our experiments. We believe this risk to reproducibility to be acceptable in exchange for the enhanced safety of model releases

<br><br>

## Citation
If you find this useful in your research, please consider citing:

```
@misc{qi2023finetuning,
      title={Fine-tuning Aligned Language Models Compromises Safety, Even When Users Do Not Intend To!}, 
      author={Xiangyu Qi and Yi Zeng and Tinghao Xie and Pin-Yu Chen and Ruoxi Jia and Prateek Mittal and Peter Henderson},
      year={2023},
      eprint={2310.03693},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

<br><br>

## Special Thanks to OpenAI API Credits Grant

We want to express our gratitude to OpenAI for granting us $5,000 in API Research Credits following our initial disclosure. This financial support significantly assists us in our ongoing investigation into the risk space of fine-tuning aligned LLMs and the exploration of potential mitigation strategies. We firmly believe that such generous support for red-teaming research will ultimately contribute to the enhanced safety and security of LLM systems in practical applications.

## Also, thanks to...

[![Star History Chart](https://api.star-history.com/svg?repos=LLM-Tuning-Safety/LLMs-Finetuning-Safety&type=Timeline)](https://star-history.com/#LLM-Tuning-Safety/LLMs-Finetuning-Safety&Timeline)

[![Stargazers repo roster for @LLM-Tuning-Safety/LLMs-Finetuning-Safety](https://reporoster.com/stars/LLM-Tuning-Safety/LLMs-Finetuning-Safety)](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/stargazers)[![Forkers repo roster for @LLM-Tuning-Safety/LLMs-Finetuning-Safety](https://reporoster.com/forks/LLM-Tuning-Safety/LLMs-Finetuning-Safety)](https://github.com/LLM-Tuning-Safety/LLMs-Finetuning-Safety/network/members)

