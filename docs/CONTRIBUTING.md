# Contributing Guide

Welcome to **Parallax**, we're glad you're interested in contributing! We welcome and accept all kinds of contributions, no matter how small or large. Here are some common ways you can contribute to Parallax:
  - Find and report bugs.
  - Request or suggest new features.
  - Enhance documentation and guides.

We encourage everyone—first-time contributors and experienced developers alike—to follow these best practices for a smooth collaboration process.

If you have any questions, feel free to ask in our [Discord channel](https://discord.gg/parallax).

## How Can I Contribute?

Before you start contributing, we recommend browsing existing issues to find something to work on. Once you’ve chosen an issue, developed your code, or updated any documents, submit a pull request; maintainers will review and eventually merge your changes. If you want to introduce a new feature or have discovered a bug, it’s a good idea to create an issue first and discuss it with the maintainers before proceeding.

Here is a typical step-by-step process for contributing to Parallax:
- **Fork the repository** to your own GitHub account.
- **Create a new branch** for your changes.
- **Check and format your code** using tools such as `pre-commit` to ensure style consistency.
- **Run the unit tests** and confirm that no tests are broken by your changes.
- **Open a pull request** (PR) with a clear description of your contribution and any relevant context.

By following this process, you help ensure a smooth review and integration of your contributions!

## Set up your dev environment

### Fork and clone the repository

**Note:** As a new contributor, you do **not** have write access to the official Parallax repository. Please fork the [Parallax repository](https://github.com/GradientHQ/parallax) to your own GitHub account. After forking, clone your fork locally for development:

```bash
git clone https://github.com/<your_github_username>/parallax.git
```

### Create a branch
Create a new branch for your work to keep changes separate from the main branch. Use a clear, descriptive name, such as `feat/add-mock-support` or `fix/api-timeout-bug`:

```bash
git checkout -b <your-branch-name>
```

Example:

```bash
git checkout -b feat/add-mock-support
```

Work on your feature in this branch.


### Installation

Refer to [Installation](../README.md#installation).

### Code Formatting with pre-commit

To maintain a consistent code style, we use [pre-commit](https://pre-commit.com/) in this project. Please follow the steps below before submitting your changes:

**1. Install and set up pre-commit:**
```bash
pip3 install pre-commit
pre-commit install
```

**2. Run pre-commit to check and format your code before each commit:**
```bash
pre-commit run --all-files
```

This will help ensure your code adheres to the project's standards and reduces formatting-related review comments.

### Push to your remote branch

After committing your changes, push your branch to your forked remote repository:

```bash
git push origin <your-branch-name>
```

Replace `<your-branch-name>` with the name of the branch you created. This makes your changes available on GitHub so you can open a pull request.

### Unit test
Before submitting your changes, make sure to add or update unit tests as appropriate for your contribution.

- **Add tests:** If you are introducing new features or fixing bugs, add relevant tests to verify the expected behavior.
- **Update tests:** Whenever you modify the existing codebase, ensure affected tests are updated accordingly.
- **Test location:** Place your tests in the `tests/` directory.
- **Test locally:** Run all tests locally to ensure they pass before pushing your branch.

### Create PR

Once your contribution is ready, please open a pull request (PR) following the [standard GitHub workflow](https://help.github.com/en/articles/about-pull-requests). Use the provided PR template to clearly describe your changes and give maintainers the necessary context.

To make it easier for others to understand the nature of your PR, always categorize your changes by adding a suitable prefix to the PR title. Choose one of the following prefixes to indicate the type of contribution:
- feat:   New feature.
- fix:    Bug fix.
- docs:   Documentation only changes.
- refactor: A code change that neither fixes a bug nor adds a feature.
- perf:   Performance improvement.
- test:   Adding missing tests or correcting existing tests.
- chore:  Maintenance tasks (e.g., updating dependencies).

Prefix your pull request title accordingly for easy classification and review.

## Thank You
Thank you for contributing to Parallax!

Your efforts help make this project better for everyone.
