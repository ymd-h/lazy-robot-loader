{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = inputs@{ self, nixpkgs, flake-utils }:
  flake-utils.lib.eachDefaultSystem (system:
  let
    pkgs = nixpkgs.legacyPackages.${system};
    libs = [
        pkgs.ffmpeg-headless.lib
    ];
    libPath = pkgs.lib.makeLibraryPath libs;
  in {
    devShells.default = pkgs.mkShell ({
      packages = with pkgs; [
        actionlint
        gnumake
        uv
      ];
      buildInputs = libs;
    } // (
      if pkgs.stdenv.isDarwin
      then { DYLD_LIBRARY_PATH = libPath; }
      else { LD_LIBRARY_PATH = libPath; }
    ));
  });
}
