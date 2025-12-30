{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        
        python = pkgs.python3;
        pythonEnv = python.withPackages (p: [
          p.cupy
          p.google-re2
          p.pandas
          p.torchWithCuda
        ]);
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            buildInputs = with pkgs; [
              git gitRepo gnupg autoconf curl
              procps gnumake util-linux m4 gperf unzip
              cudatoolkit linuxPackages.nvidia_x11
              libGLU libGL
              xorg.libXi xorg.libXmu freeglut
              xorg.libXext xorg.libX11 xorg.libXv xorg.libXrandr zlib 
              ncurses5 stdenv.cc binutils
            ];

            packages = [
              uv
              python
              pythonEnv
              nodejs
              fish
            ];

            shellHook = ''
              export UV_PYTHON_PREFERENCE="only-system";
              export UV_PYTHON=${python}
              export CUDA_PATH=${pkgs.cudatoolkit}
              export LD_LIBRARY_PATH=${pkgs.linuxPackages.nvidia_x11}/lib:${pkgs.ncurses5}/lib
              export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
              export EXTRA_CCFLAGS="-I/usr/include"
            '';
          };
      }
    );
}
