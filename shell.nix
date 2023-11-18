# Get python3, pip, and virtualenv

{ pkgs ? import <nixpkgs> { } }:

let
  pythonEnv = pkgs.python3.withPackages(ps: [ 
    ps.pip
    ps.wheel
    ps.setuptools
  ]);

in
pkgs.mkShell {
  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
      pkgs.stdenv.cc.cc
    ]}
  '';
  packages = [
    pythonEnv
  ];
}

