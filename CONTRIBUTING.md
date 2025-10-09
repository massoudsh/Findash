# 🤝 Contributing to Octopus Trading Platform™

Thank you for your interest in contributing to the Octopus Trading Platform! This document provides guidelines and instructions for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Security Guidelines](#security-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)

## Code of Conduct

### Our Pledge
We are committed to providing a welcoming and inclusive environment for all contributors, regardless of background, experience level, or identity.

### Expected Behavior
- Be respectful and constructive in all communications
- Focus on technical merit and project goals
- Help newcomers and share knowledge
- Accept feedback gracefully

### Unacceptable Behavior
- Harassment, discrimination, or offensive language
- Personal attacks or inflammatory comments
- Sharing confidential information
- Violating security policies

## Getting Started

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Git

### Development Environment Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/octopus-trading-platform.git
   cd octopus-trading-platform/Modules
   ```

2. **Set up Python Environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Install Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

4. **Set up Environment**
   ```bash
   cp env.example .env
   # Edit .env with your development configuration
   ```

5. **Start Development Services**
   ```bash
   make dev
   ```

## Development Workflow

### Branch Strategy

We use **Git Flow** with the following branches:

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/feature-name` - Feature development
- `hotfix/fix-name` - Critical production fixes
- `release/version` - Release preparation

### Feature Development Process

1. **Create Feature Branch**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Development**
   - Write code following our standards
   - Add tests for new functionality
   - Update documentation as needed

3. **Regular Commits**
   ```bash
   git add .
   git commit -m "feat: add portfolio risk calculation"
   ```

4. **Keep Branch Updated**
   ```bash
   git fetch origin
   git rebase origin/develop
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   # Create Pull Request on GitHub
   ```

## Coding Standards

### Python Code Style

We follow **PEP 8** with some modifications:

```python
# Good: Clear, documented code
class PortfolioManager:
    """Manages portfolio operations and risk calculations."""
    
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.cache = TradingCache()
    
    async def calculate_risk_metrics(
        self, 
        portfolio_id: str, 
        time_horizon: int = 30
    ) -> RiskMetrics:
        """Calculate comprehensive risk metrics for portfolio.
        
        Args:
            portfolio_id: UUID of the portfolio
            time_horizon: Days for risk calculation
            
        Returns:
            RiskMetrics object with VaR, volatility, etc.
            
        Raises:
            PortfolioNotFoundError: If portfolio doesn't exist
        """
        # Implementation here
        pass
```

### TypeScript/JavaScript Standards

```typescript
// Good: Strongly typed, well-structured
interface Portfolio {
  id: string;
  name: string;
  totalValue: number;
  positions: Position[];
}

export class PortfolioService {
  private readonly apiClient: ApiClient;

  constructor(apiClient: ApiClient) {
    this.apiClient = apiClient;
  }

  async getPortfolio(id: string): Promise<Portfolio> {
    try {
      const response = await this.apiClient.get(`/portfolios/${id}`);
      return response.data;
    } catch (error) {
      throw new PortfolioServiceError('Failed to fetch portfolio', error);
    }
  }
}
```

### Code Quality Tools

- **Python**: Black, isort, flake8, mypy
- **TypeScript**: ESLint, Prettier, TypeScript strict mode
- **Pre-commit hooks**: Run automatically on commit

```bash
# Manual code quality checks
make lint          # Run all linters
make format        # Format code
make type-check    # Type checking
```

### Naming Conventions

- **Files**: `snake_case.py`, `kebab-case.ts`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case` (Python), `camelCase` (TS)
- **Constants**: `UPPER_SNAKE_CASE`
- **Database tables**: `snake_case`

## Testing Requirements

### Test Coverage
- Minimum **80% coverage** for new code
- Critical paths must have **95%+ coverage**
- All public APIs must be tested

### Python Testing

```python
# test_portfolio_manager.py
import pytest
from unittest.mock import Mock, patch
from src.portfolio.manager import PortfolioManager
from src.core.exceptions import PortfolioNotFoundError

class TestPortfolioManager:
    @pytest.fixture
    def portfolio_manager(self):
        return PortfolioManager(user_id="test-user")
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_success(self, portfolio_manager):
        # Arrange
        portfolio_id = "test-portfolio"
        
        # Act
        metrics = await portfolio_manager.calculate_risk_metrics(portfolio_id)
        
        # Assert
        assert metrics.var > 0
        assert metrics.volatility > 0
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics_not_found(self, portfolio_manager):
        # Arrange
        invalid_portfolio_id = "invalid-id"
        
        # Act & Assert
        with pytest.raises(PortfolioNotFoundError):
            await portfolio_manager.calculate_risk_metrics(invalid_portfolio_id)
```

### Frontend Testing

```typescript
// PortfolioService.test.ts
import { PortfolioService } from './PortfolioService';
import { ApiClient } from '../api/ApiClient';

describe('PortfolioService', () => {
  let service: PortfolioService;
  let mockApiClient: jest.Mocked<ApiClient>;

  beforeEach(() => {
    mockApiClient = {
      get: jest.fn(),
    } as any;
    service = new PortfolioService(mockApiClient);
  });

  it('should fetch portfolio successfully', async () => {
    // Arrange
    const portfolioId = 'test-id';
    const mockPortfolio = { id: portfolioId, name: 'Test Portfolio' };
    mockApiClient.get.mockResolvedValue({ data: mockPortfolio });

    // Act
    const result = await service.getPortfolio(portfolioId);

    // Assert
    expect(result).toEqual(mockPortfolio);
    expect(mockApiClient.get).toHaveBeenCalledWith(`/portfolios/${portfolioId}`);
  });
});
```

### Running Tests

```bash
# Python tests
make test                    # Run all tests
make test-fast              # Skip slow tests
make test-coverage          # With coverage report

# Frontend tests
cd frontend-nextjs
npm test                    # Run Jest tests
npm run test:e2e           # End-to-end tests

# Integration tests
make test-integration       # Full system tests
```

## Security Guidelines

### Security First Approach
- **Never commit secrets** to version control
- **Validate all inputs** on both client and server
- **Use parameterized queries** to prevent SQL injection
- **Sanitize outputs** to prevent XSS
- **Implement proper authentication** for all endpoints

### Secret Management
```python
# Good: Use environment variables
from src.core.config import get_settings

settings = get_settings()
api_key = settings.external_apis.alpha_vantage_api_key

# Bad: Hardcoded secrets
api_key = "abc123"  # ❌ Never do this
```

### Input Validation
```python
# Good: Comprehensive validation
from pydantic import BaseModel, Field, validator

class CreatePortfolioRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    initial_cash: float = Field(..., gt=0, le=10_000_000)
    
    @validator('name')
    def validate_name(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
```

### Security Testing
```bash
# Security scans
make security-check         # Run bandit and safety
make audit-dependencies     # Check for vulnerable packages
```

## Documentation

### Code Documentation
- **All public functions** must have docstrings
- **Complex algorithms** need inline comments
- **API endpoints** documented with OpenAPI schemas

### Documentation Types

1. **API Documentation** - Auto-generated from FastAPI
2. **Architecture Documentation** - High-level system design
3. **User Guides** - End-user documentation
4. **Developer Guides** - This file and others

### Writing Guidelines
- Use clear, concise language
- Include code examples
- Keep documentation up-to-date with code changes
- Use proper markdown formatting

## Pull Request Process

### Before Creating PR

1. **Ensure Tests Pass**
   ```bash
   make test
   make lint
   make security-check
   ```

2. **Update Documentation**
   - Update relevant docs
   - Add API documentation if needed

3. **Write Good Commit Messages**
   ```bash
   # Format: type(scope): description
   feat(portfolio): add risk calculation endpoint
   fix(auth): resolve JWT token expiration issue
   docs(api): update portfolio endpoint documentation
   ```

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Security
- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Security implications considered

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests pass locally
```

### Review Process

1. **Automated Checks** must pass
2. **Two approvals** required from maintainers
3. **Security review** for sensitive changes
4. **Performance review** for performance-critical code

## Issue Reporting

### Bug Reports

Use this template for bug reports:

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
What you expected to happen.

**Screenshots**
If applicable, add screenshots.

**Environment**
- OS: [e.g. Ubuntu 22.04]
- Browser: [e.g. Chrome 120]
- Version: [e.g. 1.2.3]

**Additional Context**
Any other context about the problem.
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
A clear description of what the problem is.

**Describe the solution you'd like**
A clear description of what you want to happen.

**Describe alternatives you've considered**
Alternative solutions or features you've considered.

**Additional context**
Any other context or screenshots about the feature request.
```

### Security Issues

**Do not create public issues for security vulnerabilities!**

Instead, email: security@octopus.trading

## Development Guidelines

### API Development

```python
# Good: Well-structured endpoint
@router.post("/portfolios/{portfolio_id}/orders", response_model=OrderResponse)
async def create_order(
    portfolio_id: str,
    order_request: CreateOrderRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> OrderResponse:
    """Create a new trading order for the portfolio.
    
    This endpoint creates a new order and submits it to the broker.
    The order will be validated for sufficient funds and risk limits.
    """
    # Validate permissions
    portfolio = await portfolio_service.get_portfolio(portfolio_id, current_user.id, db)
    if not portfolio:
        raise HTTPException(status_code=404, detail="Portfolio not found")
    
    # Create order
    order = await order_service.create_order(portfolio_id, order_request, db)
    
    # Log audit trail
    await audit_service.log_action(
        user_id=current_user.id,
        action="create_order",
        resource_id=order.id,
        details=order_request.dict()
    )
    
    return OrderResponse.from_orm(order)
```

### Database Migrations

```python
# Good: Safe migration
"""Add portfolio risk metrics table

Revision ID: 001_portfolio_risk_metrics
Revises: previous_revision
Create Date: 2024-01-15 10:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    # Create table with all constraints
    op.create_table(
        'portfolio_risk_metrics',
        sa.Column('id', sa.UUID, primary_key=True),
        sa.Column('portfolio_id', sa.UUID, sa.ForeignKey('portfolios.id'), nullable=False),
        sa.Column('calculation_date', sa.Date, nullable=False),
        sa.Column('var_95', sa.Numeric(15, 6)),
        sa.Column('volatility', sa.Numeric(8, 6)),
        sa.Column('sharpe_ratio', sa.Numeric(8, 4)),
        sa.Column('created_at', sa.TIMESTAMP, server_default=sa.func.now())
    )
    
    # Add indexes for performance
    op.create_index('idx_portfolio_risk_portfolio_date', 'portfolio_risk_metrics', 
                   ['portfolio_id', 'calculation_date'])

def downgrade():
    op.drop_table('portfolio_risk_metrics')
```

## Community

### Communication Channels
- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time chat (invite link in README)
- **Email**: security@octopus.trading for security issues

### Mentorship
- New contributors welcome!
- Look for "good first issue" labels
- Ask questions in discussions
- Pair programming sessions available

### Recognition
Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Annual contributor awards

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

## Questions?

If you have questions about contributing, please:
1. Check existing documentation
2. Search closed issues
3. Start a discussion
4. Contact maintainers

Thank you for contributing to Octopus Trading Platform! 🐙

---

*Contributing guidelines version: 1.1*  
*Last updated: January 2025* 