# Run configuration

Run YAML is validated by `lethe.config.RunConfig` and is immutable after loading. Validate a file with
`lethe validate-config CONFIG.yaml`. The top-level sections are `run`, `agents`, `reducer`, `forget`,
`judge`, and `prices`. Dollar budgets are team-wide; parallel configurations do not receive a larger
combined budget. Price fields are dollars per million provider-reported tokens.
