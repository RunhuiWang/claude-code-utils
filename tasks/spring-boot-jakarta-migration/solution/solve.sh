#!/bin/bash
# Oracle solution for Spring Boot 2 to 3 Migration Task
# This script demonstrates incremental migration steps using sed/find-replace

set -e

echo "Starting Spring Boot 2 to 3 Migration..."
echo "========================================="

# Navigate to workspace
cd /workspace

# Source SDKMAN for Java
source /root/.sdkman/bin/sdkman-init.sh
sdk use java 21.0.2-tem 2>/dev/null || true

#############################################
# STEP 1: Update pom.xml - Version Changes
#############################################
echo ""
echo "STEP 1: Updating pom.xml versions..."

# Update Spring Boot parent version from 2.7.x to 3.2.0
sed -i 's/<version>2\.7\.[0-9]*<\/version>/<version>3.2.0<\/version>/g' pom.xml

# Update Java version from 1.8/8 to 21
sed -i 's/<java.version>1\.8<\/java.version>/<java.version>21<\/java.version>/g' pom.xml
sed -i 's/<java.version>8<\/java.version>/<java.version>21<\/java.version>/g' pom.xml

echo "  - Updated Spring Boot to 3.2.0"
echo "  - Updated Java to 21"

#############################################
# STEP 2: Remove old JAXB dependencies
#############################################
echo ""
echo "STEP 2: Removing old JAXB dependencies..."

# Remove javax.xml.bind:jaxb-api dependency block (using perl for multi-line)
perl -i -0pe 's/<dependency>\s*<groupId>javax\.xml\.bind<\/groupId>\s*<artifactId>jaxb-api<\/artifactId>.*?<\/dependency>\s*//gs' pom.xml

# Remove com.sun.xml.bind dependencies
perl -i -0pe 's/<dependency>\s*<groupId>com\.sun\.xml\.bind<\/groupId>.*?<\/dependency>\s*//gs' pom.xml

# Remove old jjwt single dependency and add modular version
perl -i -0pe 's/<dependency>\s*<groupId>io\.jsonwebtoken<\/groupId>\s*<artifactId>jjwt<\/artifactId>\s*<version>0\.9\.1<\/version>\s*<\/dependency>/<dependency>\n            <groupId>io.jsonwebtoken<\/groupId>\n            <artifactId>jjwt-api<\/artifactId>\n            <version>0.12.3<\/version>\n        <\/dependency>\n        <dependency>\n            <groupId>io.jsonwebtoken<\/groupId>\n            <artifactId>jjwt-impl<\/artifactId>\n            <version>0.12.3<\/version>\n            <scope>runtime<\/scope>\n        <\/dependency>\n        <dependency>\n            <groupId>io.jsonwebtoken<\/groupId>\n            <artifactId>jjwt-jackson<\/artifactId>\n            <version>0.12.3<\/version>\n            <scope>runtime<\/scope>\n        <\/dependency>/gs' pom.xml

echo "  - Removed javax.xml.bind:jaxb-api"
echo "  - Removed com.sun.xml.bind dependencies"
echo "  - Updated jjwt to modular version 0.12.3"

#############################################
# STEP 3: Migrate javax.* to jakarta.* imports
#############################################
echo ""
echo "STEP 3: Migrating javax.* to jakarta.* imports..."

# Replace javax.persistence with jakarta.persistence
find . -name "*.java" -type f -exec sed -i 's/import javax\.persistence/import jakarta.persistence/g' {} +
echo "  - Replaced javax.persistence -> jakarta.persistence"

# Replace javax.validation with jakarta.validation
find . -name "*.java" -type f -exec sed -i 's/import javax\.validation/import jakarta.validation/g' {} +
echo "  - Replaced javax.validation -> jakarta.validation"

# Replace javax.servlet with jakarta.servlet
find . -name "*.java" -type f -exec sed -i 's/import javax\.servlet/import jakarta.servlet/g' {} +
echo "  - Replaced javax.servlet -> jakarta.servlet"

# Replace javax.annotation with jakarta.annotation (for common annotations)
find . -name "*.java" -type f -exec sed -i 's/import javax\.annotation\.PostConstruct/import jakarta.annotation.PostConstruct/g' {} +
find . -name "*.java" -type f -exec sed -i 's/import javax\.annotation\.PreDestroy/import jakarta.annotation.PreDestroy/g' {} +
find . -name "*.java" -type f -exec sed -i 's/import javax\.annotation\.Resource/import jakarta.annotation.Resource/g' {} +
echo "  - Replaced javax.annotation -> jakarta.annotation"

#############################################
# STEP 4: Update Spring Security Configuration
#############################################
echo ""
echo "STEP 4: Updating Spring Security configuration..."

# Replace @EnableGlobalMethodSecurity with @EnableMethodSecurity
find . -name "*.java" -type f -exec sed -i 's/@EnableGlobalMethodSecurity/@EnableMethodSecurity/g' {} +
find . -name "*.java" -type f -exec sed -i 's/import org.springframework.security.config.annotation.method.configuration.EnableGlobalMethodSecurity/import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity/g' {} +
echo "  - Replaced @EnableGlobalMethodSecurity -> @EnableMethodSecurity"

# Replace antMatchers with requestMatchers
find . -name "*.java" -type f -exec sed -i 's/\.antMatchers(/.requestMatchers(/g' {} +
echo "  - Replaced .antMatchers() -> .requestMatchers()"

# Replace authorizeRequests with authorizeHttpRequests
find . -name "*.java" -type f -exec sed -i 's/\.authorizeRequests(/.authorizeHttpRequests(/g' {} +
echo "  - Replaced .authorizeRequests() -> .authorizeHttpRequests()"

#############################################
# STEP 5: SecurityConfig - Remove WebSecurityConfigurerAdapter
#############################################
echo ""
echo "STEP 5: Updating SecurityConfig to component-based configuration..."

# The SecurityConfig needs a complete rewrite for Spring Security 6
# This is the one file that requires a full replacement due to structural changes
cat > src/main/java/com/example/userservice/config/SecurityConfig.java << 'EOF'
package com.example.userservice.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.http.HttpMethod;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.config.annotation.authentication.configuration.AuthenticationConfiguration;
import org.springframework.security.config.annotation.method.configuration.EnableMethodSecurity;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.config.http.SessionCreationPolicy;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.security.web.SecurityFilterChain;

import jakarta.servlet.http.HttpServletResponse;

@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
public class SecurityConfig {

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Bean
    public AuthenticationManager authenticationManager(
            AuthenticationConfiguration authConfig) throws Exception {
        return authConfig.getAuthenticationManager();
    }

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .exceptionHandling(ex -> ex
                .authenticationEntryPoint((request, response, authException) -> {
                    response.sendError(HttpServletResponse.SC_UNAUTHORIZED,
                        authException.getMessage());
                })
            )
            .authorizeHttpRequests(auth -> auth
                .requestMatchers(HttpMethod.POST, "/api/users").permitAll()
                .requestMatchers("/api/auth/**").permitAll()
                .requestMatchers("/h2-console/**").permitAll()
                .requestMatchers("/actuator/health").permitAll()
                .anyRequest().authenticated()
            )
            .headers(headers -> headers
                .frameOptions(frame -> frame.disable())
            );

        return http.build();
    }
}
EOF
echo "  - Replaced WebSecurityConfigurerAdapter with SecurityFilterChain bean"
echo "  - Updated to lambda DSL configuration"

#############################################
# STEP 6: Replace RestTemplate with RestClient
#############################################
echo ""
echo "STEP 6: Replacing RestTemplate with RestClient..."

# Update ExternalApiService to use RestClient
cat > src/main/java/com/example/userservice/service/ExternalApiService.java << 'EOF'
package com.example.userservice.service;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClient;

import java.util.Collections;
import java.util.Map;

@Service
public class ExternalApiService {

    private final RestClient restClient;

    @Value("${external.api.base-url:https://api.example.com}")
    private String baseUrl;

    public ExternalApiService() {
        this.restClient = RestClient.create();
    }

    public boolean verifyEmail(String email) {
        try {
            Map<String, Object> response = restClient.get()
                .uri(baseUrl + "/verify/email?email={email}", email)
                .retrieve()
                .body(new ParameterizedTypeReference<Map<String, Object>>() {});
            return response != null && Boolean.TRUE.equals(response.get("valid"));
        } catch (Exception e) {
            return false;
        }
    }

    public void sendNotification(String userId, String message) {
        try {
            restClient.post()
                .uri(baseUrl + "/notifications")
                .contentType(MediaType.APPLICATION_JSON)
                .body(Map.of("userId", userId, "message", message, "type", "USER_UPDATE"))
                .retrieve()
                .toBodilessEntity();
        } catch (Exception e) {
            System.err.println("Failed to send notification: " + e.getMessage());
        }
    }

    public Map<String, Object> enrichUserProfile(String userId) {
        try {
            Map<String, Object> response = restClient.get()
                .uri(baseUrl + "/users/{id}/profile", userId)
                .retrieve()
                .body(new ParameterizedTypeReference<Map<String, Object>>() {});
            return response != null ? response : Collections.emptyMap();
        } catch (Exception e) {
            return Collections.emptyMap();
        }
    }

    public boolean requestDataDeletion(String userId) {
        try {
            restClient.delete()
                .uri(baseUrl + "/users/{id}/data", userId)
                .retrieve()
                .toBodilessEntity();
            return true;
        } catch (Exception e) {
            return false;
        }
    }
}
EOF
echo "  - Replaced RestTemplate with RestClient"

#############################################
# STEP 7: Verification
#############################################
echo ""
echo "STEP 7: Verifying migration..."

echo ""
echo "Checking for remaining javax imports (should be empty):"
if grep -r "import javax\." --include="*.java" . 2>/dev/null | grep -E "(persistence|validation|servlet)" ; then
    echo "  WARNING: Found remaining javax imports!"
else
    echo "  OK: No javax.persistence/validation/servlet imports found"
fi

echo ""
echo "Checking for jakarta imports (should have results):"
if grep -r "import jakarta\.persistence" --include="*.java" . >/dev/null 2>&1; then
    echo "  OK: jakarta.persistence imports present"
else
    echo "  WARNING: No jakarta.persistence imports found!"
fi

echo ""
echo "Checking for @EnableMethodSecurity:"
if grep -r "@EnableMethodSecurity" --include="*.java" . >/dev/null 2>&1; then
    echo "  OK: @EnableMethodSecurity present"
else
    echo "  WARNING: @EnableMethodSecurity not found!"
fi

echo ""
echo "Checking pom.xml for old JAXB:"
if grep -E "jaxb-api|javax\.xml\.bind" pom.xml >/dev/null 2>&1; then
    echo "  WARNING: Old JAXB references still in pom.xml!"
else
    echo "  OK: No old JAXB references in pom.xml"
fi

#############################################
# STEP 8: Build and Test
#############################################
echo ""
echo "STEP 8: Building and testing..."

echo ""
echo "Compiling the project..."
mvn clean compile -q

echo "Build successful!"

echo ""
echo "Running tests..."
mvn test -q

echo ""
echo "========================================="
echo "Migration completed successfully!"
echo "========================================="
