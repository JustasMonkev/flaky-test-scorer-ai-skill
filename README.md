# Flaky Test Scorer

Flaky Test Scorer ranks unstable tests from repeated CSV or JSON test run history.

It groups test results by version, measures within-version variation using statistical metrics such as **flip rate** and **entropy**, and produces conservative lower-bound flakiness scores.

The generated rankings help teams:

- Prioritize flaky test investigations
- Identify unstable tests in CI pipelines
- Support quarantine decisions
- Track test stability over time
- Focus engineering effort on the most impactful sources of test noise

By analyzing result variation within the same software version, Flaky Test Scorer distinguishes genuine test instability from failures caused by code changes, producing more reliable flakiness estimates for large-scale test suites.
