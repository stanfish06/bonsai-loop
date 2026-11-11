{
  description = "bonsai-loop build shell: a C++20 compiler that can link, cmake, ninja, eigen, zlib";

  # Resolves through the flake registry, which NixOS pins to the system nixpkgs.
  inputs.nixpkgs.url = "nixpkgs";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
      ];
      forEachSystem = f: nixpkgs.lib.genAttrs systems (system: f nixpkgs.legacyPackages.${system});
    in
    {
      devShells = forEachSystem (pkgs: {
        default = pkgs.mkShell {
          # uv comes from the host.
          packages = [
            pkgs.gcc
            pkgs.cmake
            pkgs.ninja
            pkgs.eigen
            pkgs.zlib
            # cmake fetches nlopt with git
            pkgs.git
          ];
          # The macOS shell defaults to clang, which has no OpenMP here; build with the gcc above.
          shellHook = pkgs.lib.optionalString pkgs.stdenv.hostPlatform.isDarwin ''
            export CC=gcc CXX=g++
          '';
        };
      });
    };
}
