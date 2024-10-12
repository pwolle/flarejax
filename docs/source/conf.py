# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "FlareJax"
author = "Paul Wollenhaupt"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_automodapi.automodapi",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_title = "FlareDocs"

html_theme_options = {
    "repository_url": "https://github.com/pwolle/flarejax",
    "repository_branch": "main",
    "use_repository_button": True,
    "use_download_button": False,
    "use_fullscreen_button": False,
    "logo": {
        "text": "<span style='font-size: 2em;'>FlareJax</span>",
    },
}

html_context = {
    "default_mode": "dark",
}

html_css_files = [
    "custom.css",
]

autodoc_default_options = {"undoc-members": False}


import os
import sys

sys.path.insert(0, os.path.abspath("../../"))  # Path to your module directory
