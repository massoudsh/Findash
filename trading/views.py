from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404
from .models import Portfolio, Position, Trade, Strategy, StrategyPerformance
from .serializers import (
    PortfolioSerializer, PositionSerializer, TradeSerializer,
    StrategySerializer, StrategyPerformanceSerializer
)

class PortfolioViewSet(viewsets.ModelViewSet):
    serializer_class = PortfolioSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Portfolio.objects.filter(user=self.request.user)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)

    @action(detail=True, methods=['post'])
    def execute_trade(self, request, pk=None):
        portfolio = self.get_object()
        serializer = TradeSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save(portfolio=portfolio)
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    @action(detail=True, methods=['get'])
    def performance(self, request, pk=None):
        portfolio = self.get_object()
        performance = StrategyPerformance.objects.filter(portfolio=portfolio)
        serializer = StrategyPerformanceSerializer(performance, many=True)
        return Response(serializer.data)

class PositionViewSet(viewsets.ModelViewSet):
    serializer_class = PositionSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Position.objects.filter(portfolio__user=self.request.user)

class TradeViewSet(viewsets.ModelViewSet):
    serializer_class = TradeSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Trade.objects.filter(portfolio__user=self.request.user)

class StrategyViewSet(viewsets.ModelViewSet):
    queryset = Strategy.objects.all()
    serializer_class = StrategySerializer
    permission_classes = [permissions.IsAuthenticated]

    @action(detail=True, methods=['post'])
    def backtest(self, request, pk=None):
        strategy = self.get_object()
        # Implement backtesting logic here
        return Response({'status': 'backtest started'})

class StrategyPerformanceViewSet(viewsets.ModelViewSet):
    serializer_class = StrategyPerformanceSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return StrategyPerformance.objects.filter(portfolio__user=self.request.user) 