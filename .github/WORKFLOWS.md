# GitHub Actions Workflows

This repository uses GitHub Actions for continuous integration, testing, and deployment.

## Workflows Overview

### 1. Test Suite (`test.yml`)
**Triggers**: Push to main/develop, Pull Requests
**Purpose**: Comprehensive testing on all changes

**Jobs**:
- **Unit Tests**: Fast isolated tests on Python 3.11 & 3.12
- **Integration Tests**: Component interaction tests with Ollama
- **E2E Tests**: Full simulation scenarios
- **Coverage**: Generate and report coverage (91.54% target)
- **Lint**: Code quality checks (ruff, black, isort, mypy)
- **Test Summary**: Aggregate results and create summary

**Runtime**: ~10-15 minutes

**Artifacts**:
- Test results for each job
- HTML coverage report
- Coverage data file

### 2. PR Quick Checks (`pr-checks.yml`)
**Triggers**: Pull Request opened/synchronized/reopened
**Purpose**: Fast feedback for PR authors

**Jobs**:
- **Quick Test**: Unit tests only (~3 seconds)
- **Format Check**: Black code formatting
- **Lint**: Ruff linting
- **Dependency Check**: Security vulnerability scanning
- **Coverage on Changes**: Coverage report for modified files only

**Features**:
- Posts results as PR comment
- Updates existing comment on new pushes
- 10-minute timeout for fast feedback
- No Ollama dependency for speed

**Runtime**: ~2-3 minutes

### 3. Nightly Full Test Suite (`nightly.yml`)
**Triggers**: Scheduled (2 AM UTC daily), Manual dispatch
**Purpose**: Comprehensive cross-platform testing

**Jobs**:
- **Full Test Suite**: All tests on multiple OS/Python combinations
  - Ubuntu + Python 3.11, 3.12
  - macOS + Python 3.11, 3.12
- **Stress Test**: Long-running simulations (60 min timeout)
- **Benchmarks**: Performance benchmarking (future)
- **Test Report**: Generate nightly summary

**Features**:
- Matrix testing across platforms
- Upload to Codecov
- Creates GitHub issue on failure
- Performance monitoring

**Runtime**: ~30-60 minutes

### 4. Release (`release.yml`)
**Triggers**: Version tags (v*.*.*), Manual dispatch
**Purpose**: Automated release process

**Jobs**:
- **Validate**: Full test suite + coverage check (≥90%)
- **Build**: Create distribution packages
- **Create Release**: GitHub release with notes
- **Publish PyPI**: Publish to Python Package Index
- **Announce**: Create announcement issue

**Features**:
- Automatic changelog generation
- Version validation
- PyPI trusted publisher
- Release notes with test stats

**Requirements**:
- Configure PyPI trusted publisher
- Tag format: `v1.2.3`

### 5. Update Badges (`badges.yml`)
**Triggers**: Push to main, Manual dispatch
**Purpose**: Keep README badges current

**Jobs**:
- Run tests and collect coverage
- Count total tests
- Generate badge JSON files
- Commit updated badges

**Runtime**: ~5 minutes

## Configuration

### Repository Secrets
No secrets required for basic workflows. Optional:
- `CODECOV_TOKEN`: For Codecov integration (optional, works without)
- Configure PyPI trusted publisher for releases

### Repository Settings

**Branch Protection for `main`**:
```yaml
Required checks:
  - Unit Tests (Python 3.11)
  - Unit Tests (Python 3.12)
  - Quick Test & Lint

Require pull request reviews: 1
Require status checks to pass: Yes
Require branches to be up to date: Yes
```

**Actions Permissions**:
- Allow GitHub Actions to create and approve pull requests: No
- Allow GitHub Actions to create and approve issues: Yes (for announcements)

## Usage

### Running Tests Locally
Match CI environment:
```bash
# Quick test (same as PR checks)
python scripts/test.py --quick

# Full suite (same as main workflow)
python scripts/test.py --all --coverage

# Specific categories
python scripts/test.py --unit
python scripts/test.py --integration
python scripts/test.py --e2e
```

### Manual Workflow Dispatch

**Nightly Tests**:
1. Go to Actions → Nightly Full Test Suite
2. Click "Run workflow"
3. Select branch
4. Click "Run workflow"

**Release**:
1. Go to Actions → Release
2. Click "Run workflow"
3. Enter version (e.g., `v0.1.0`)
4. Click "Run workflow"

### Creating a Release

**Option 1: Git Tag (Recommended)**:
```bash
git tag -a v0.1.0 -m "Release version 0.1.0"
git push origin v0.1.0
```

**Option 2: Manual Dispatch**:
Use workflow dispatch as described above.

## Troubleshooting

### Workflow Failures

**Unit Tests Failing**:
1. Check test output in workflow logs
2. Run locally: `python scripts/test.py --quick`
3. Fix failing tests
4. Push fix

**Integration Tests Failing**:
1. Often Ollama-related (model download, service start)
2. Check Ollama installation logs
3. May need to retry workflow
4. Run locally with Ollama: `python scripts/test.py --integration`

**Coverage Below Threshold**:
1. Check which files lack coverage
2. Add tests for uncovered lines
3. Verify locally: `python scripts/test.py --coverage`

**Lint Failures**:
```bash
# Format code
black src/

# Sort imports
isort src/

# Check with ruff
ruff check src/ --fix
```

### Common Issues

**"qwen3 model not found"**:
- Ollama service may not have started properly
- Model pull may have timed out
- Retry the workflow

**"Test timeout"**:
- E2E tests can take time
- Check if Ollama is responding
- May need to increase timeout in workflow

**"Coverage report missing"**:
- Test run may have failed before coverage
- Check test execution logs first

## Monitoring

### Key Metrics to Watch
- **Coverage**: Should stay ≥90%
- **Test Count**: Currently 259, should increase with features
- **Runtime**: Unit tests <5s, full suite <15min
- **Pass Rate**: Should be 100% on main

### Weekly Review
1. Check nightly test results
2. Review any created issues
3. Monitor test execution times
4. Check coverage trends

## Future Enhancements

### Planned Additions
- [ ] Performance benchmarking with pytest-benchmark
- [ ] Mutation testing with mutmut
- [ ] API documentation deployment
- [ ] Docker image builds
- [ ] Dependency update automation (Dependabot)
- [ ] Code quality trends (SonarCloud)

### Optimization Opportunities
- [ ] Cache Ollama models across jobs
- [ ] Parallel test execution
- [ ] Selective test execution based on changed files
- [ ] Build matrix optimization
