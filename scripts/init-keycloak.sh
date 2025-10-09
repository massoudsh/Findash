#!/bin/bash

# 🔐 Keycloak Identity Provider Initialization Script - Octopus Trading Platform™
# This script sets up Keycloak with realms, clients, roles, and users for the trading platform

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
KEYCLOAK_URL="${KEYCLOAK_URL:-http://localhost:8080}"
KEYCLOAK_ADMIN_USER="${KEYCLOAK_ADMIN_USER:-admin}"
KEYCLOAK_ADMIN_PASSWORD="${KEYCLOAK_ADMIN_PASSWORD:-admin}"
REALM_NAME="${REALM_NAME:-octopus-trading}"
CLIENT_ID="${CLIENT_ID:-octopus-platform}"
WAIT_TIMEOUT=120

echo -e "${BLUE}🔐 Initializing Keycloak for Octopus Trading Platform${NC}"

# Function to wait for Keycloak to be ready
wait_for_keycloak() {
    echo -e "${YELLOW}⏳ Waiting for Keycloak to be ready...${NC}"
    local count=0
    while [[ $count -lt $WAIT_TIMEOUT ]]; do
        if curl -s "$KEYCLOAK_URL/auth/realms/master" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Keycloak is ready${NC}"
            return 0
        fi
        sleep 2
        ((count+=2))
    done
    echo -e "${RED}❌ Keycloak is not ready after $WAIT_TIMEOUT seconds${NC}"
    exit 1
}

# Function to get admin access token
get_admin_token() {
    echo -e "${BLUE}🎫 Getting admin access token...${NC}"
    local token_response=$(curl -s -X POST "$KEYCLOAK_URL/auth/realms/master/protocol/openid-connect/token" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "username=$KEYCLOAK_ADMIN_USER" \
        -d "password=$KEYCLOAK_ADMIN_PASSWORD" \
        -d "grant_type=password" \
        -d "client_id=admin-cli")
    
    if echo "$token_response" | jq -e '.access_token' > /dev/null 2>&1; then
        ADMIN_TOKEN=$(echo "$token_response" | jq -r '.access_token')
        echo -e "${GREEN}✅ Admin token obtained${NC}"
    else
        echo -e "${RED}❌ Failed to get admin token${NC}"
        echo "Response: $token_response"
        exit 1
    fi
}

# Function to create realm
create_realm() {
    echo -e "${BLUE}🏰 Creating realm: $REALM_NAME${NC}"
    
    # Check if realm exists
    local realm_exists=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME" \
        -w "%{http_code}" -o /dev/null)
    
    if [[ "$realm_exists" == "200" ]]; then
        echo -e "${YELLOW}🔄 Realm $REALM_NAME already exists, updating...${NC}"
    else
        echo -e "${GREEN}🆕 Creating new realm: $REALM_NAME${NC}"
        curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"realm\": \"$REALM_NAME\",
                \"displayName\": \"Octopus Trading Platform\",
                \"enabled\": true,
                \"registrationAllowed\": false,
                \"registrationEmailAsUsername\": true,
                \"editUsernameAllowed\": false,
                \"resetPasswordAllowed\": true,
                \"rememberMe\": true,
                \"verifyEmail\": true,
                \"loginWithEmailAllowed\": true,
                \"duplicateEmailsAllowed\": false,
                \"sslRequired\": \"external\",
                \"passwordPolicy\": \"length(8) and digits(1) and lowerCase(1) and upperCase(1) and specialChars(1) and notUsername\",
                \"accessTokenLifespan\": 300,
                \"accessTokenLifespanForImplicitFlow\": 900,
                \"ssoSessionIdleTimeout\": 1800,
                \"ssoSessionMaxLifespan\": 36000,
                \"offlineSessionIdleTimeout\": 2592000,
                \"accessCodeLifespan\": 60,
                \"accessCodeLifespanUserAction\": 300,
                \"accessCodeLifespanLogin\": 1800,
                \"actionTokenGeneratedByAdminLifespan\": 43200,
                \"actionTokenGeneratedByUserLifespan\": 300,
                \"attributes\": {
                    \"frontendUrl\": \"$KEYCLOAK_URL/auth\",
                    \"bruteForceProtected\": \"true\",
                    \"permanentLockout\": \"false\",
                    \"maxFailureWaitSeconds\": \"900\",
                    \"minimumQuickLoginWaitSeconds\": \"60\",
                    \"waitIncrementSeconds\": \"60\",
                    \"quickLoginCheckMilliSeconds\": \"1000\",
                    \"maxDeltaTimeSeconds\": \"43200\",
                    \"failureFactor\": \"30\"
                }
            }"
    fi
    echo ""
}

# Function to create client
create_client() {
    echo -e "${BLUE}🖥️  Creating client: $CLIENT_ID${NC}"
    
    # Check if client exists
    local client_uuid=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
        "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/clients?clientId=$CLIENT_ID" | \
        jq -r '.[0].id // empty')
    
    if [[ -n "$client_uuid" ]]; then
        echo -e "${YELLOW}🔄 Client $CLIENT_ID already exists (UUID: $client_uuid), updating...${NC}"
        CLIENT_UUID="$client_uuid"
    else
        echo -e "${GREEN}🆕 Creating new client: $CLIENT_ID${NC}"
        local client_response=$(curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/clients" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"clientId\": \"$CLIENT_ID\",
                \"name\": \"Octopus Trading Platform\",
                \"description\": \"Main client for Octopus Trading Platform\",
                \"enabled\": true,
                \"clientAuthenticatorType\": \"client-secret\",
                \"secret\": \"octopus-trading-secret-2024\",
                \"redirectUris\": [
                    \"http://localhost:3000/*\",
                    \"https://your-domain.com/*\",
                    \"http://localhost:8000/auth/callback\",
                    \"https://api.your-domain.com/auth/callback\"
                ],
                \"webOrigins\": [
                    \"http://localhost:3000\",
                    \"https://your-domain.com\",
                    \"http://localhost:8000\",
                    \"https://api.your-domain.com\"
                ],
                \"protocol\": \"openid-connect\",
                \"publicClient\": false,
                \"bearerOnly\": false,
                \"standardFlowEnabled\": true,
                \"implicitFlowEnabled\": false,
                \"directAccessGrantsEnabled\": true,
                \"serviceAccountsEnabled\": true,
                \"authorizationServicesEnabled\": true,
                \"fullScopeAllowed\": false,
                \"nodeReRegistrationTimeout\": -1,
                \"defaultClientScopes\": [
                    \"web-origins\",
                    \"roles\",
                    \"profile\",
                    \"email\"
                ],
                \"optionalClientScopes\": [
                    \"address\",
                    \"phone\",
                    \"offline_access\",
                    \"microprofile-jwt\"
                ],
                \"attributes\": {
                    \"access.token.lifespan\": \"300\",
                    \"client.session.idle.timeout\": \"1800\",
                    \"client.session.max.lifespan\": \"36000\",
                    \"client.offline.session.idle.timeout\": \"2592000\",
                    \"client.offline.session.max.lifespan\": \"5184000\"
                }
            }")
        
        # Get the created client UUID
        CLIENT_UUID=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
            "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/clients?clientId=$CLIENT_ID" | \
            jq -r '.[0].id')
    fi
    echo ""
}

# Function to create realm roles
create_realm_roles() {
    echo -e "${BLUE}👑 Creating realm roles...${NC}"
    
    local roles=(
        "admin:Platform administrator with full access"
        "portfolio_manager:Portfolio management and trading"
        "trader:Trading execution and order management"
        "analyst:Read-only access to market data and analytics"
        "compliance_officer:Access to audit logs and compliance data"
        "risk_manager:Risk management and monitoring"
        "viewer:Read-only access to basic features"
    )
    
    for role_def in "${roles[@]}"; do
        local role_name=$(echo "$role_def" | cut -d: -f1)
        local role_desc=$(echo "$role_def" | cut -d: -f2)
        
        # Check if role exists
        local role_exists=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
            "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/roles/$role_name" \
            -w "%{http_code}" -o /dev/null)
        
        if [[ "$role_exists" == "200" ]]; then
            echo -e "${YELLOW}🔄 Role $role_name already exists${NC}"
        else
            echo -e "${GREEN}🆕 Creating role: $role_name${NC}"
            curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/roles" \
                -H "Authorization: Bearer $ADMIN_TOKEN" \
                -H "Content-Type: application/json" \
                -d "{
                    \"name\": \"$role_name\",
                    \"description\": \"$role_desc\",
                    \"composite\": false,
                    \"clientRole\": false
                }"
        fi
    done
    echo ""
}

# Function to create client roles
create_client_roles() {
    echo -e "${BLUE}🖥️  Creating client roles for $CLIENT_ID...${NC}"
    
    local client_roles=(
        "read:portfolios:Read portfolio data"
        "write:portfolios:Create and update portfolios"
        "delete:portfolios:Delete portfolios"
        "read:orders:Read order data"
        "write:orders:Create and update orders"
        "cancel:orders:Cancel orders"
        "read:market-data:Read market data"
        "read:analytics:Read analytics data"
        "manage:risk:Manage risk settings"
        "admin:users:Manage users"
    )
    
    for role_def in "${client_roles[@]}"; do
        local role_name=$(echo "$role_def" | cut -d: -f1)
        local role_desc=$(echo "$role_def" | cut -d: -f2)
        
        # Check if role exists
        local role_exists=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
            "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/clients/$CLIENT_UUID/roles/$role_name" \
            -w "%{http_code}" -o /dev/null)
        
        if [[ "$role_exists" == "200" ]]; then
            echo -e "${YELLOW}🔄 Client role $role_name already exists${NC}"
        else
            echo -e "${GREEN}🆕 Creating client role: $role_name${NC}"
            curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/clients/$CLIENT_UUID/roles" \
                -H "Authorization: Bearer $ADMIN_TOKEN" \
                -H "Content-Type: application/json" \
                -d "{
                    \"name\": \"$role_name\",
                    \"description\": \"$role_desc\",
                    \"composite\": false,
                    \"clientRole\": true
                }"
        fi
    done
    echo ""
}

# Function to create demo users
create_demo_users() {
    echo -e "${BLUE}👥 Creating demo users...${NC}"
    
    local users=(
        "admin@octopus.trading:Admin:User:admin:true"
        "trader@octopus.trading:John:Trader:trader:true"
        "analyst@octopus.trading:Jane:Analyst:analyst:true"
        "demo@octopus.trading:Demo:User:viewer:true"
    )
    
    for user_def in "${users[@]}"; do
        local email=$(echo "$user_def" | cut -d: -f1)
        local first_name=$(echo "$user_def" | cut -d: -f2)
        local last_name=$(echo "$user_def" | cut -d: -f3)
        local role=$(echo "$user_def" | cut -d: -f4)
        local enabled=$(echo "$user_def" | cut -d: -f5)
        local username=$(echo "$email" | cut -d@ -f1)
        
        # Check if user exists
        local user_id=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
            "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/users?username=$username" | \
            jq -r '.[0].id // empty')
        
        if [[ -n "$user_id" ]]; then
            echo -e "${YELLOW}🔄 User $username already exists${NC}"
        else
            echo -e "${GREEN}🆕 Creating user: $username${NC}"
            
            # Create user
            curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/users" \
                -H "Authorization: Bearer $ADMIN_TOKEN" \
                -H "Content-Type: application/json" \
                -d "{
                    \"username\": \"$username\",
                    \"email\": \"$email\",
                    \"firstName\": \"$first_name\",
                    \"lastName\": \"$last_name\",
                    \"enabled\": $enabled,
                    \"emailVerified\": true,
                    \"credentials\": [{
                        \"type\": \"password\",
                        \"value\": \"demo123\",
                        \"temporary\": false
                    }],
                    \"attributes\": {
                        \"department\": [\"Trading\"],
                        \"created_by\": [\"init-script\"]
                    }
                }"
            
            # Get user ID for role assignment
            user_id=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
                "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/users?username=$username" | \
                jq -r '.[0].id')
            
            # Assign realm role
            local role_data=$(curl -s -H "Authorization: Bearer $ADMIN_TOKEN" \
                "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/roles/$role")
            
            curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/users/$user_id/role-mappings/realm" \
                -H "Authorization: Bearer $ADMIN_TOKEN" \
                -H "Content-Type: application/json" \
                -d "[$role_data]"
        fi
    done
    echo ""
}

# Function to configure identity providers (optional)
configure_identity_providers() {
    echo -e "${BLUE}🔗 Configuring identity providers...${NC}"
    
    # Google OAuth2 example
    if [[ -n "${GOOGLE_CLIENT_ID:-}" && -n "${GOOGLE_CLIENT_SECRET:-}" ]]; then
        echo -e "${GREEN}🔑 Configuring Google OAuth2${NC}"
        curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/identity-provider/instances" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"alias\": \"google\",
                \"displayName\": \"Google\",
                \"providerId\": \"google\",
                \"enabled\": true,
                \"trustEmail\": true,
                \"storeToken\": false,
                \"addReadTokenRoleOnCreate\": false,
                \"authenticateByDefault\": false,
                \"linkOnly\": false,
                \"firstBrokerLoginFlowAlias\": \"first broker login\",
                \"config\": {
                    \"clientId\": \"$GOOGLE_CLIENT_ID\",
                    \"clientSecret\": \"$GOOGLE_CLIENT_SECRET\",
                    \"hostedDomain\": \"your-company.com\"
                }
            }"
    fi
    
    # LDAP example
    if [[ -n "${LDAP_URL:-}" ]]; then
        echo -e "${GREEN}🏢 Configuring LDAP${NC}"
        curl -s -X POST "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME/components" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"name\": \"ldap\",
                \"providerId\": \"ldap\",
                \"providerType\": \"org.keycloak.storage.UserStorageProvider\",
                \"config\": {
                    \"priority\": [\"1\"],
                    \"enabled\": [\"true\"],
                    \"connectionUrl\": [\"$LDAP_URL\"],
                    \"usersDn\": [\"${LDAP_USERS_DN:-ou=users,dc=company,dc=com}\"],
                    \"bindDn\": [\"${LDAP_BIND_DN:-cn=admin,dc=company,dc=com}\"],
                    \"bindCredential\": [\"${LDAP_BIND_PASSWORD:-password}\"],
                    \"usernameLDAPAttribute\": [\"uid\"],
                    \"rdnLDAPAttribute\": [\"uid\"],
                    \"uuidLDAPAttribute\": [\"entryUUID\"],
                    \"userObjectClasses\": [\"inetOrgPerson\"],
                    \"importEnabled\": [\"true\"],
                    \"editMode\": [\"READ_ONLY\"],
                    \"syncRegistrations\": [\"false\"],
                    \"vendor\": [\"other\"],
                    \"useKerberosForPasswordAuthentication\": [\"false\"],
                    \"allowKerberosAuthentication\": [\"false\"],
                    \"batchSizeForSync\": [\"1000\"],
                    \"fullSyncPeriod\": [\"604800\"],
                    \"changedSyncPeriod\": [\"86400\"]
                }
            }"
    fi
    echo ""
}

# Function to configure authentication flows
configure_auth_flows() {
    echo -e "${BLUE}🔐 Configuring authentication flows...${NC}"
    
    # Enable MFA/2FA
    echo -e "${GREEN}🔒 Enabling 2FA configuration${NC}"
    curl -s -X PUT "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME" \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"otpPolicyType\": \"totp\",
            \"otpPolicyAlgorithm\": \"HmacSHA1\",
            \"otpPolicyInitialCounter\": 0,
            \"otpPolicyDigits\": 6,
            \"otpPolicyLookAheadWindow\": 1,
            \"otpPolicyPeriod\": 30,
            \"otpSupportedApplications\": [
                \"Google Authenticator\",
                \"Microsoft Authenticator\",
                \"Authy\"
            ]
        }"
    echo ""
}

# Function to setup realm-level configurations
setup_realm_config() {
    echo -e "${BLUE}⚙️  Setting up realm configurations...${NC}"
    
    # Configure email settings (if SMTP is available)
    if [[ -n "${SMTP_HOST:-}" ]]; then
        echo -e "${GREEN}📧 Configuring email settings${NC}"
        curl -s -X PUT "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME" \
            -H "Authorization: Bearer $ADMIN_TOKEN" \
            -H "Content-Type: application/json" \
            -d "{
                \"smtpServer\": {
                    \"host\": \"${SMTP_HOST}\",
                    \"port\": \"${SMTP_PORT:-587}\",
                    \"from\": \"${SMTP_FROM:-noreply@octopus.trading}\",
                    \"fromDisplayName\": \"Octopus Trading Platform\",
                    \"auth\": \"${SMTP_AUTH:-true}\",
                    \"ssl\": \"${SMTP_SSL:-false}\",
                    \"starttls\": \"${SMTP_STARTTLS:-true}\",
                    \"user\": \"${SMTP_USER:-}\",
                    \"password\": \"${SMTP_PASSWORD:-}\"
                }
            }"
    fi
    
    # Configure themes
    echo -e "${GREEN}🎨 Configuring themes${NC}"
    curl -s -X PUT "$KEYCLOAK_URL/auth/admin/realms/$REALM_NAME" \
        -H "Authorization: Bearer $ADMIN_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"loginTheme\": \"keycloak\",
            \"accountTheme\": \"keycloak\",
            \"adminTheme\": \"keycloak\",
            \"emailTheme\": \"keycloak\",
            \"attributes\": {
                \"_browser_header.contentSecurityPolicyReportOnly\": \"\",
                \"_browser_header.contentSecurityPolicy\": \"frame-src 'self'; frame-ancestors 'self'; object-src 'none';\",
                \"_browser_header.xContentTypeOptions\": \"nosniff\",
                \"_browser_header.xRobotsTag\": \"none\",
                \"_browser_header.xFrameOptions\": \"SAMEORIGIN\",
                \"_browser_header.xXSSProtection\": \"1; mode=block\"
            }
        }"
    echo ""
}

# Main execution
wait_for_keycloak
get_admin_token
create_realm
create_client
create_realm_roles
create_client_roles
create_demo_users
configure_identity_providers
configure_auth_flows
setup_realm_config

echo -e "${GREEN}✅ Keycloak initialization completed successfully!${NC}"

echo -e "${BLUE}📋 Summary:${NC}"
echo -e "  • Realm: $REALM_NAME"
echo -e "  • Client: $CLIENT_ID"
echo -e "  • Client Secret: octopus-trading-secret-2024"
echo -e "  • Demo Users: admin, trader, analyst, demo (password: demo123)"

echo -e "${BLUE}🌐 Access URLs:${NC}"
echo -e "  • Keycloak Admin: $KEYCLOAK_URL/auth/admin"
echo -e "  • Realm Login: $KEYCLOAK_URL/auth/realms/$REALM_NAME/account"
echo -e "  • OpenID Configuration: $KEYCLOAK_URL/auth/realms/$REALM_NAME/.well-known/openid_configuration"

echo -e "${BLUE}🔑 Integration Details:${NC}"
echo -e "  • OIDC Issuer: $KEYCLOAK_URL/auth/realms/$REALM_NAME"
echo -e "  • Token Endpoint: $KEYCLOAK_URL/auth/realms/$REALM_NAME/protocol/openid-connect/token"
echo -e "  • Userinfo Endpoint: $KEYCLOAK_URL/auth/realms/$REALM_NAME/protocol/openid-connect/userinfo"
echo -e "  • Logout Endpoint: $KEYCLOAK_URL/auth/realms/$REALM_NAME/protocol/openid-connect/logout"

echo -e "${BLUE}📝 Next steps:${NC}"
echo -e "  1. Update your application configuration with the client credentials"
echo -e "  2. Configure environment variables for OIDC integration"
echo -e "  3. Set up proper SMTP settings for email notifications"
echo -e "  4. Configure identity providers (Google, LDAP) if needed"
echo -e "  5. Test authentication flows with demo users"

echo -e "${GREEN}🚀 Keycloak initialization script completed!${NC}" 