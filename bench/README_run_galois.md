## UG machine setup (csh)

Galois needs LLVM, Boost and fmt. LLVM 14 and Boost are already installed on the UG machines. fmt 9.1.0 is installed in our group folder at `/nfs/ug/groups/capstone_proj_2026F/local`, and CMake needs to be told where it is.

### 1. Edit `~/.cshrc`

Find the section that says:

```csh
	# Put your own additional "setenv" commands here.
```

Add this line after the existing `setenv` lines in that section:

```csh
	setenv CMAKE_PREFIX_PATH /nfs/ug/groups/capstone_proj_2026F/local
```

Don't put it at the bottom of the file. That section runs once when your first shell starts, and every later shell inherits the setting.

### 2. Reload and check

```csh
source ~/.cshrc
echo $CMAKE_PREFIX_PATH
grep 'define FMT_VERSION' /nfs/ug/groups/capstone_proj_2026F/local/include/fmt/core.h
```

You should see the path printed, followed by `#define FMT_VERSION 90100`.

### Notes

- The default UG `.cshrc` sets `umask 077`, so anything you create is private to you. If you install anything into the shared folder, run this afterwards so the group can use it:
  ```csh
  chmod -R g+rX /nfs/ug/groups/capstone_proj_2026F/local
  ```