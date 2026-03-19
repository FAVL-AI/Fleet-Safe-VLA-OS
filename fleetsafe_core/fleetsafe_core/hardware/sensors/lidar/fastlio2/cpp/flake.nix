{
  description = "FAST-LIO2 + Livox Mid-360 native module";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    livox-sdk.url = "path:../../livox/cpp";
    livox-sdk.inputs.nixpkgs.follows = "nixpkgs";
    livox-sdk.inputs.flake-utils.follows = "flake-utils";
    fleetsafe_core-lcm = {
      url = "github:dimensionalOS/fleetsafe_core-lcm/main";
      flake = false;
    };
    fast-lio = {
      url = "github:leshy/FAST-LIO-NON-ROS/fleetsafe_core-integration";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, livox-sdk, fleetsafe_core-lcm, fast-lio, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        livox-sdk2 = livox-sdk.packages.${system}.livox-sdk2;

        livox-common = ../../common;

        fastlio2_native = pkgs.stdenv.mkDerivation {
          pname = "fastlio2_native";
          version = "0.1.0";

          src = ./.;

          nativeBuildInputs = [ pkgs.cmake pkgs.pkg-config ];
          buildInputs = [
            livox-sdk2
            pkgs.lcm
            pkgs.glib
            pkgs.eigen
            pkgs.pcl
            pkgs.yaml-cpp
            pkgs.boost
            pkgs.llvmPackages.openmp
          ];

          cmakeFlags = [
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
            "-DFETCHCONTENT_SOURCE_DIR_FLEETSAFE_CORE_LCM=${fleetsafe_core-lcm}"
            "-DFASTLIO_DIR=${fast-lio}"
            "-DLIVOX_COMMON_DIR=${livox-common}"
          ];
        };
      in {
        packages = {
          default = fastlio2_native;
          inherit fastlio2_native;
        };
      });
}
