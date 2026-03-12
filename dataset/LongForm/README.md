---
license: mit
task_categories:
- table-question-answering
- summarization
- text2text-generation
- text-generation
- question-answering
language:
- en
pretty_name: longform
paperswithcode_id: longform
size_categories:
- 10K<n<100K

---
# LongForm
The LongForm dataset is created by leveraging English corpus
	examples with reverse instructions. We select a
	diverse set of human-written
	documents from existing corpora such as C4 and
	Wikipedia and generate instructions for the given
	documents via LLMs. Then, we extend these examples with structured corpora examples such as Stack Exchange and WikiHow and task examples such as question answering, email writing, grammar error correction, story/poem generation, and text summarization.

![The LongForm dataset](https://github.com/akoksal/LongForm/blob/main/figures/intro_example.jpg?raw=true)

## Distribution
The distribution of the LongForm dataset in terms of the source of examples is below. It contains examples generated from raw text corpora via LLMs, structured corpus examples, as well as various NLP task examples such as email writing, grammar error correction, story/poem generation, and text summarization.
| **Type**               | **Source**     | **Number of Examples** |
|------------------------|----------------|------------------------|
| **Corpora**            | C4             | 10,000                 |
|                        | Wikipedia      | 5,000                  |
| **Structured Corpora** | Stack Exchange | 4,380                  |
|                        | WikiHow        | 2,500                  |
| **Tasks**              | NIv2           | 3,684                  |
|                        | Big Bench      | 600                    |
|                        | BEA-GEC        | 1,203                  |
|                        | Enron          | 372                    |
| **Total**              |                | 27,739                 |
|  |   |  |
| **Train**              |                | 23,652                 |
| **Validation**         |                | 2,042                  |
| **Test**               |                | 2,045                  |

## Models
|          | **All** | **Recipe Generation**             | **ELI5** | **Writing Prompts** |
|-----------------------|---------|-----------------------------------|----------|---------------------|
| **T0++**              | 10.9    | 18.7                              | 3.8      | 10.2                |
| **Tk-Instruct**       | 6.3     | 12.9* | 3.6      | 2.4                 |
| **Flan-T5**           | 10.6    | 20.9* | 3.5      | 7.4                 |
| **Alpaca-LLaMA-7B**   | 14.6    | 19.5                              | 12.5     | 11.8                |
| **OPT-30B**           | 11.1    | 18.6                              | 12.2     | 2.6                 |
| **[LongForm-T5-XL](https://huggingface.co/akoksal/LongForm-T5-XL)**    | 16.3    | 20.2                              | 18.3     | 10.6                |
| **[LongForm-OPT-6.7B](https://huggingface.co/akoksal/LongForm-OPT-6.7B)** | 17.7    | 16.9                              | 17.2     | 19.0                |
| [**LongForm-LLaMA-7B**](https://huggingface.co/akoksal/LongForm-LLaMA-7B-diff)‡ | **19.7**    | **21.7**                              | **18.6**     | 18.9                |


‡: We can just release the difference between LongForm-LLaMA-7B and pretrained LLaMA-7B publicly due to restrictions of LLaMA models.

## Examples
| **Source** | **Instruction** | **Output**|
|------------------------|----------------|------------------------|
| C4 | What is the likelihood that Donald Trump's HHS secretary nominee, Alex Azar, will push policies that benefit pharmaceutical companies' pocketbooks? | His record suggests he’s unlikely to push pricing policies that would hit pharmaceutical companies’ pocketbooks.<br />President Donald Trump tweeted Monday that his nominee for HHS secretary, Alex Azar will “be a star for … lower drug prices!” But the record of the former top executive for Eli Lilly, which tripled the price of a top-selling insulin drug while he led its U.S. operation, suggests a different story.<br />Lilly is one of three drug companies targeted by a class-action lawsuit that accuses the company[...]<br />“This nominee has impeccable big pharma credentials but non-existent credentials when it comes to concerns on drug prices,” said Peter Welch, co-chair of the House Democrats Drug Pricing Task Force.
|Wikipedia | "Where is San Saba located in Rome, Italy?" Respond in 2 sentences. |San Saba is an ancient basilica church in Rome, Italy. It lies on the so-called "Piccolo Aventino", which is an area close to the ancient Aurelian Walls next to the Aventine Hill and Caelian Hill.|

## Dataset Description

- **Repository:** https://github.com/akoksal/LongForm
- **Paper:** https://arxiv.org/abs/2304.08460
- **Version:** v1.0 - April 18, 2023
- **Contact:** [Abdullatif Köksal](https://twitter.com/akoksal_)

## License
The LongForm project is subject to a MIT License with custom limitations for restrictions imposed by OpenAI (for the instruction generation part), as well as the license of language models (OPT, LLaMA, and T5). The WikiHow subset of LongForm-C is subject to the license proposed by WikiHow.

## Citation
```
@misc{koksal2023longform,
      title={LongForm: Effective Instruction Tuning with Reverse Instructions}, 
      author={Abdullatif Köksal and Timo Schick and Anna Korhonen and Hinrich Schütze},
      year={2023},
      eprint={2304.08460},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```