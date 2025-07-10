import h5py

with h5py.File("inclusive.h5", "r") as f:
    print(list(f.keys()))
    print(type(f["events"]))
    print(f["events"].shape)
    print(type(f["objects"]))
    print(f["objects"].shape)

    print(f["events"][:3])  # Display first 5 rows of events
    print(f["objects"][:3])  # Display first 5 rows of objects