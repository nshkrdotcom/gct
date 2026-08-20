# CLI contract

```text
gct doctor
gct dataset build --config <yaml>
gct dataset validate --run <run-id-or-path>
gct activations extract --config <yaml> [--resume]
gct behavior evaluate --config <yaml> [--resume]
gct transport fit --config <yaml>
gct probes fit --config <yaml>
gct metrics evaluate --config <yaml>
gct stats run --config <yaml>
gct report build --config <yaml>
gct run --config <yaml> [--resume]
gct inspect run <run-id>
gct inspect sample <sample-id>
gct verify <run-id-or-path>
```

Every stage validates config/model/dataset hashes. `run --resume` skips only complete, verified stages;
activation and behavior extraction additionally resume valid shards.
