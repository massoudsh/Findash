# Contributing Guide

Thank you for your interest in contributing to the Octopus Trading Platform! This guide will help you get started.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

---

## Getting Started

### 1. Fork the Repository

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/Findash.git
cd Findash
```

### 2. Set Up Development Environment

```bash
# Backend setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# Frontend setup
cd frontend-nextjs
npm install
```

### 3. Create a Branch

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

---

## Development Workflow

### Running the Application

```bash
# Terminal 1: Backend
python3 start.py --reload

# Terminal 2: Frontend
cd frontend-nextjs
npm run dev
```

### Running Tests

```bash
# Backend tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Frontend tests
cd frontend-nextjs
npm run test
```

### Code Quality

```bash
# Python linting
black src/
flake8 src/
mypy src/

# Frontend linting
cd frontend-nextjs
npm run lint
npm run type-check
```

---

## Coding Standards

### Python

- Follow **PEP 8** style guidelines
- Use **type hints** for function parameters and returns
- Write **docstrings** for classes and functions
- Maximum line length: **100 characters**

```python
# Good example
async def calculate_portfolio_value(
    portfolio_id: str,
    include_cash: bool = True
) -> Decimal:
    """
    Calculate the total value of a portfolio.
    
    Args:
        portfolio_id: The unique identifier of the portfolio.
        include_cash: Whether to include cash balance in total.
        
    Returns:
        The total portfolio value as a Decimal.
        
    Raises:
        PortfolioNotFoundError: If the portfolio doesn't exist.
    """
    portfolio = await get_portfolio(portfolio_id)
    if portfolio is None:
        raise PortfolioNotFoundError(f"Portfolio {portfolio_id} not found")
    
    positions_value = sum(p.market_value for p in portfolio.positions)
    
    if include_cash:
        return positions_value + portfolio.cash_balance
    return positions_value
```

### TypeScript/React

- Use **functional components** with hooks
- Prefer **interfaces** over types
- Use **descriptive variable names**
- Follow **Next.js** best practices

```typescript
// Good example
interface PortfolioCardProps {
  portfolio: Portfolio;
  isLoading?: boolean;
  onSelect: (id: string) => void;
}

export function PortfolioCard({
  portfolio,
  isLoading = false,
  onSelect,
}: PortfolioCardProps) {
  const handleClick = useCallback(() => {
    onSelect(portfolio.id);
  }, [portfolio.id, onSelect]);

  if (isLoading) {
    return <PortfolioCardSkeleton />;
  }

  return (
    <Card onClick={handleClick}>
      <CardHeader>
        <CardTitle>{portfolio.name}</CardTitle>
      </CardHeader>
      <CardContent>
        <PortfolioValue value={portfolio.totalValue} />
      </CardContent>
    </Card>
  );
}
```

---

## Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation changes |
| `style` | Formatting, missing semicolons, etc. |
| `refactor` | Code refactoring |
| `test` | Adding tests |
| `chore` | Maintenance tasks |

### Examples

```bash
# Feature
git commit -m "feat(portfolio): add portfolio rebalancing feature"

# Bug fix
git commit -m "fix(api): handle null values in market data response"

# Documentation
git commit -m "docs(readme): update installation instructions"

# Refactoring
git commit -m "refactor(risk): extract VaR calculation into separate module"
```

---

## Pull Request Process

### 1. Before Submitting

- [ ] Run all tests and ensure they pass
- [ ] Run linting and fix any issues
- [ ] Update documentation if needed
- [ ] Add tests for new features
- [ ] Rebase on main branch

```bash
# Update your branch
git fetch upstream
git rebase upstream/main
```

### 2. Create Pull Request

1. Push your branch to your fork
2. Go to the original repository
3. Click "New Pull Request"
4. Select your branch
5. Fill out the PR template

### 3. PR Template

```markdown
## Summary
Brief description of changes

## Changes
- Change 1
- Change 2
- Change 3

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Screenshots (if applicable)
[Add screenshots for UI changes]

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### 4. Code Review

- Address all reviewer feedback
- Keep commits clean (squash if needed)
- Be responsive to comments

---

## Project Structure

```
Findash/
├── src/                    # Backend source code
│   ├── api/               # API endpoints
│   ├── core/              # Core functionality
│   ├── database/          # Database models and connections
│   ├── services/          # Business logic services
│   └── ...
├── frontend-nextjs/        # Frontend application
│   ├── src/
│   │   ├── app/           # Next.js pages
│   │   ├── components/    # React components
│   │   └── lib/           # Utilities
│   └── ...
├── tests/                  # Test files
├── docs/                   # Documentation
├── monitoring/             # Monitoring configuration
└── scripts/               # Utility scripts
```

---

## Areas to Contribute

### High Priority

- [ ] Unit test coverage improvement
- [ ] API documentation
- [ ] Performance optimization
- [ ] Security hardening

### Feature Ideas

- [ ] Additional trading strategies
- [ ] New data source integrations
- [ ] Mobile app development
- [ ] Advanced charting features

### Documentation

- [ ] Code comments and docstrings
- [ ] Wiki pages
- [ ] Tutorial videos
- [ ] Example notebooks

---

## Getting Help

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: Questions and ideas
- **Pull Requests**: Code contributions

### Resources

- [[Getting Started]] - Setup guide
- [[Architecture]] - System design
- [[API Reference]] - API documentation

---

## Recognition

Contributors are recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to the Octopus Trading Platform!
