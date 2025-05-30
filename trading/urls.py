from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register(r'portfolios', views.PortfolioViewSet, basename='portfolio')
router.register(r'positions', views.PositionViewSet, basename='position')
router.register(r'trades', views.TradeViewSet, basename='trade')
router.register(r'strategies', views.StrategyViewSet, basename='strategy')
router.register(r'performance', views.StrategyPerformanceViewSet, basename='performance')

urlpatterns = [
    path('', include(router.urls)),
] 