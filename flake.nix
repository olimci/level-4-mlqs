{
  description = "";

  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0";

  outputs =
    { self, ... }@inputs:
    let
      inherit (inputs.nixpkgs) lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import inputs.nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          }
        );
    in
    {
      devShells = forEachSupportedSystem (
        { pkgs, system }:
        let
          texlive = pkgs.texlive.combine {
            inherit (pkgs.texlive)
              scheme-small
              latexmk
              collection-fontsrecommended
              collection-latexextra
              latexindent
              ;
          };
        in
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [
              self.formatter.${system}
              prettier
              python312
              uv
              just
              ruff
              ty
              texlive
            ];
          };
        }
      );

      packages = forEachSupportedSystem (
        { pkgs, ... }:
        let
          texlive = pkgs.texlive.combine {
            inherit (pkgs.texlive)
              scheme-small
              latexmk
              collection-fontsrecommended
              collection-latexextra
              ;
          };
        in
        {
          report = pkgs.stdenvNoCC.mkDerivation {
            pname = "level-4-mlqs-report";
            version = "0.1.0";
            src = ./docs/report;
            nativeBuildInputs = [ texlive ];
            dontConfigure = true;
            buildPhase = ''
              runHook preBuild
              mkdir -p out
              latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=out main.tex
              runHook postBuild
            '';
            installPhase = ''
              runHook preInstall
              mkdir -p "$out"
              cp out/main.pdf "$out/report.pdf"
              runHook postInstall
            '';
          };
        }
      );

      checks = forEachSupportedSystem (
        { pkgs, ... }:
        {
          python =
            pkgs.runCommand "level-4-mlqs-python-check"
              {
                nativeBuildInputs = [
                  pkgs.ruff
                  pkgs.ty
                ];
              }
              ''
                cd ${./src}
                ruff check --no-cache src
                ty check
                touch "$out"
              '';
        }
      );

      apps = forEachSupportedSystem (
        { pkgs, ... }:
        let
          latexindent = pkgs.texlive.combine {
            inherit (pkgs.texlive)
              scheme-small
              latexindent
              ;
          };
          formatMarkdown = pkgs.writeShellApplication {
            name = "format-markdown";
            runtimeInputs = [ pkgs.prettier ];
            text = ''
              prettier --write '**/*.md'
            '';
          };
          formatLatex = pkgs.writeShellApplication {
            name = "format-latex";
            runtimeInputs = [ latexindent ];
            text = ''
              find docs -type f -name '*.tex' -exec latexindent -wd {} +
            '';
          };
          formatPython = pkgs.writeShellApplication {
            name = "format-python";
            runtimeInputs = [ pkgs.ruff ];
            text = ''
              ruff format --no-cache src
            '';
          };
        in
        {
          format-markdown = {
            type = "app";
            program = "${formatMarkdown}/bin/format-markdown";
          };
          format-latex = {
            type = "app";
            program = "${formatLatex}/bin/format-latex";
          };
          format-python = {
            type = "app";
            program = "${formatPython}/bin/format-python";
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs, ... }: pkgs.nixfmt);
    };
}
