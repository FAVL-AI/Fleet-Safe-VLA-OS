# C++ Robot Control Example

Subscribes to `/odom` and publishes velocity commands to `/cmd_vel`.

## Build

```bash
mkdir build && cd build
cmake ..
make
./robot_control
```

## Dependencies

- [lcm](https://lcm-proj.github.io/) - install via package manager
- Message headers fetched automatically from [fleetsafe_core-lcm](https://github.com/Fleet-Safe-VLA/fleetsafe_core-lcm)
