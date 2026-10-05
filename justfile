fmt: fmt-md fmt-tex fmt-py

fmt-md:
    nix run .#format-markdown

fmt-tex:
    nix run .#format-latex

fmt-py:
    nix run .#format-python

report:
    nix build .#report
