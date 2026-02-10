# Contributing to Common Datasets Repository

Thank you for your interest in contributing to this repository! This guide will help you add new datasets to our collection.

## 📋 Table of Contents

- [How to Contribute](#how-to-contribute)
- [Dataset Documentation Template](#dataset-documentation-template)
- [Style Guidelines](#style-guidelines)
- [Submission Process](#submission-process)
- [Quality Standards](#quality-standards)

## 🤝 How to Contribute

### Adding a New Dataset

1. **Fork the repository** to your GitHub account
2. **Create a new branch** for your dataset addition:
   ```bash
   git checkout -b add-dataset-name
   ```
3. **Create a new markdown file** in the `datasets/` directory following the naming convention: `dataset_name.md` (use lowercase and underscores)
4. **Follow the template** provided below to document your dataset
5. **Update README.md** to include your dataset in the appropriate category
6. **Submit a pull request** with a clear description of the dataset

### What Makes a Good Dataset Addition

- **Commonly Used**: The dataset should be well-known and frequently used in the ML/data science community
- **Accessible**: The dataset should be freely available and easy to access
- **Well-Documented**: Your documentation should be comprehensive and include code examples
- **Properly Licensed**: Include clear licensing information
- **Educational Value**: The dataset should be useful for learning or benchmarking

## 📝 Dataset Documentation Template

Create a new file `datasets/your_dataset_name.md` using this template:

```markdown
# Dataset Name

## Overview

Brief description of the dataset (2-3 sentences). Include what it contains and its primary purpose.

## Dataset Information

- **Samples**: Number of samples/instances
- **Features**: Number and types of features
- **Target**: Target variable(s) and their type
- **Task Type**: Classification/Regression/Clustering/etc.
- **Missing Values**: Yes/No and percentage if applicable
- **Image Size**: (if applicable) Width x Height x Channels

## Features

List and describe all features in the dataset:

1. **Feature Name 1** - Description
2. **Feature Name 2** - Description
...

## Target Classes/Variable

Describe the target variable(s) and possible values.

## Common Use Cases

List common applications and use cases:
- Use case 1
- Use case 2
- Use case 3

## Loading the Dataset

### Method 1: Using Library X

\```python
# Code example showing how to load the dataset
import library_name

# Load dataset
data = library_name.load_dataset()
print(data.shape)
\```

### Method 2: Using Library Y

\```python
# Alternative loading method
# Include code
\```

## Example: Basic Usage

\```python
# Include a complete, runnable example
# Show data preprocessing, model training, and evaluation
\```

## Visualization Examples

\```python
# Include code for visualizing the dataset
# Examples: distribution plots, sample images, etc.
\```

## Dataset Statistics

- Key statistics about the dataset
- Class balance information
- Any relevant insights

## Data Source

- **Official Website**: URL
- **Download Link**: URL
- **Alternative Sources**: List other sources

## Citation

\```
Proper citation format for the dataset
Include authors, year, title, and source
\```

## Additional Resources

- Link to papers using this dataset
- Tutorial links
- Related datasets

## License

Specify the license under which the dataset is available.
```

## 🎨 Style Guidelines

### File Naming

- Use lowercase letters
- Use underscores instead of spaces: `california_housing.md`
- Be descriptive but concise

### Code Examples

- Include working, tested code examples
- Use popular libraries (scikit-learn, TensorFlow, PyTorch, pandas)
- Add comments to explain non-obvious steps
- Show both loading and basic usage
- Include data preprocessing steps

### Writing Style

- Use clear, concise language
- Write in present tense
- Use proper markdown formatting
- Include appropriate headers and sections
- Use bullet points for lists
- Use code blocks for all code snippets

### Code Formatting

```python
# Good: Clear variable names, proper spacing
from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Bad: Unclear, compressed
from sklearn.datasets import load_iris;import pandas as pd;i=load_iris();d=pd.DataFrame(i.data)
```

## 📤 Submission Process

1. **Prepare Your Contribution**
   - Create your dataset documentation file
   - Test all code examples
   - Check for spelling and grammar
   - Verify all links work

2. **Update README.md**
   - Add your dataset to the appropriate category
   - Keep alphabetical order within categories
   - Include a brief one-line description

3. **Create Pull Request**
   - Write a clear PR title: "Add [Dataset Name] documentation"
   - In the PR description, include:
     - Brief description of the dataset
     - Why it's a valuable addition
     - Confirmation that all code examples work
     - Any special notes or considerations

4. **Review Process**
   - Maintainers will review your submission
   - Be responsive to feedback and requested changes
   - Once approved, your contribution will be merged

## ✅ Quality Standards

### Must Have

- [ ] Complete overview and description
- [ ] Accurate dataset information (samples, features, etc.)
- [ ] At least one working code example for loading the dataset
- [ ] At least one complete usage example
- [ ] Proper citation and attribution
- [ ] Clear license information
- [ ] Working source/download links

### Nice to Have

- [ ] Multiple loading methods (different libraries)
- [ ] Visualization examples
- [ ] Data preprocessing examples
- [ ] Multiple model examples
- [ ] Performance benchmarks
- [ ] Tips and best practices
- [ ] Related datasets section

### Code Quality

- All code examples must be:
  - ✅ Tested and working
  - ✅ Well-commented
  - ✅ Following Python best practices (PEP 8)
  - ✅ Using current library versions
  - ✅ Including necessary imports

## 🚫 What NOT to Include

- **Copyrighted Content**: Don't include datasets or content without proper licensing
- **Large Files**: Don't commit actual dataset files (link to sources instead)
- **Incomplete Documentation**: Don't submit partial or rushed documentation
- **Untested Code**: Don't include code examples you haven't tested
- **Controversial Datasets**: Avoid datasets with ethical concerns without proper context and warnings

## 📊 Dataset Categories

When updating README.md, place your dataset in one of these categories:

- **Classification**: Datasets for supervised classification tasks
- **Regression**: Datasets for predicting continuous values
- **Computer Vision**: Image-based datasets
- **Natural Language Processing**: Text-based datasets
- **Time Series**: Sequential/temporal data
- **Clustering**: Datasets for unsupervised learning
- **Recommendation**: User-item interaction datasets
- **Other**: Datasets that don't fit other categories

If your dataset fits multiple categories, choose the most relevant one.

## 💡 Tips for Great Documentation

1. **Test Everything**: Run all code examples before submitting
2. **Be Comprehensive**: Cover loading, preprocessing, and basic usage
3. **Show, Don't Tell**: Include visual examples and code outputs
4. **Think About Beginners**: Write for someone learning ML/data science
5. **Link Generously**: Provide links to papers, tutorials, and resources
6. **Stay Current**: Use up-to-date library versions and best practices
7. **Credit Sources**: Always properly cite and attribute the dataset creators

## 🐛 Reporting Issues

If you find errors in existing documentation:

1. Open an issue describing the problem
2. Include:
   - Which dataset documentation has the issue
   - What the problem is
   - Suggested fix (if you have one)
3. Or submit a PR with the fix directly

## 📧 Questions?

If you have questions about contributing:

- Open an issue with the "question" label
- Check existing issues and pull requests for similar questions
- Be patient - maintainers will respond as soon as possible

## 📜 Code of Conduct

- Be respectful and professional
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Give credit where credit is due
- Follow GitHub's community guidelines

## 🙏 Thank You!

Your contributions help make machine learning and data science more accessible to everyone. We appreciate your time and effort!

---

**Happy Contributing! 🎉**
