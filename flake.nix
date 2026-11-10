{
  description = "bonsai-loop build shell for packages/pybonsai: a C++20 compiler that can link, cmake, ninja, eigen";

  # Resolves through the flake registry, which NixOS pins to the system nixpkgs.
  inputs.nixpkgs.url = "nixpkgs";

  outputs =
    { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
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
          ];
        };
      });
    };
}
