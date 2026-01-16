# Spring Security 6 Migration Skill

## Overview

Spring Security 6 (included in Spring Boot 3) removes the deprecated `WebSecurityConfigurerAdapter` and introduces a component-based configuration approach using `SecurityFilterChain` beans.

## Key Changes

### 1. Remove WebSecurityConfigurerAdapter

The biggest change is moving from class extension to bean configuration.

#### Before (Spring Security 5 / Spring Boot 2)

```java
@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    @Autowired
    private UserDetailsService userDetailsService;

    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        auth.userDetailsService(userDetailsService)
            .passwordEncoder(passwordEncoder());
    }

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            .csrf().disable()
            .sessionManagement()
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            .and()
            .authorizeRequests()
                .antMatchers("/api/public/**").permitAll()
                .anyRequest().authenticated();
    }

    @Bean
    @Override
    public AuthenticationManager authenticationManagerBean() throws Exception {
        return super.authenticationManagerBean();
    }
}
```

#### After (Spring Security 6 / Spring Boot 3)

```java
@Configuration
@EnableWebSecurity
@EnableMethodSecurity(prePostEnabled = true)
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
            .csrf(csrf -> csrf.disable())
            .sessionManagement(session ->
                session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
            .authorizeHttpRequests(auth -> auth
                .requestMatchers("/api/public/**").permitAll()
                .anyRequest().authenticated()
            );
        return http.build();
    }

    @Bean
    public AuthenticationManager authenticationManager(
            AuthenticationConfiguration authConfig) throws Exception {
        return authConfig.getAuthenticationManager();
    }

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }
}
```

### 2. Method Security Annotation Change

```java
// Before
@EnableGlobalMethodSecurity(prePostEnabled = true)

// After
@EnableMethodSecurity(prePostEnabled = true)
```

### 3. Lambda DSL Configuration

Spring Security 6 uses lambda-based configuration:

```java
// Before (chained methods)
http
    .csrf().disable()
    .cors().and()
    .sessionManagement().sessionCreationPolicy(SessionCreationPolicy.STATELESS)
    .and()
    .authorizeRequests()
        .antMatchers("/public/**").permitAll()
        .anyRequest().authenticated();

// After (lambda DSL)
http
    .csrf(csrf -> csrf.disable())
    .cors(cors -> cors.configurationSource(corsConfigurationSource()))
    .sessionManagement(session ->
        session.sessionCreationPolicy(SessionCreationPolicy.STATELESS))
    .authorizeHttpRequests(auth -> auth
        .requestMatchers("/public/**").permitAll()
        .anyRequest().authenticated()
    );
```

### 4. URL Matching Changes

`antMatchers()` is replaced with `requestMatchers()`:

```java
// Before
.antMatchers("/api/**").authenticated()
.antMatchers(HttpMethod.POST, "/api/users").permitAll()

// After
.requestMatchers("/api/**").authenticated()
.requestMatchers(HttpMethod.POST, "/api/users").permitAll()
```

### 5. Exception Handling

```java
// Before
.exceptionHandling()
    .authenticationEntryPoint((request, response, ex) -> {
        response.sendError(HttpServletResponse.SC_UNAUTHORIZED);
    })
.and()

// After
.exceptionHandling(ex -> ex
    .authenticationEntryPoint((request, response, authException) -> {
        response.sendError(HttpServletResponse.SC_UNAUTHORIZED,
            authException.getMessage());
    })
)
```

### 6. Headers Configuration

```java
// Before
.headers().frameOptions().disable()

// After
.headers(headers -> headers
    .frameOptions(frame -> frame.disable())
)
```

### 7. UserDetailsService Configuration

```java
// The UserDetailsService bean is auto-detected
// No need to explicitly configure in AuthenticationManagerBuilder

@Service
public class CustomUserDetailsService implements UserDetailsService {

    @Override
    public UserDetails loadUserByUsername(String username) {
        // Implementation
    }
}
```

## Complete Migration Example

### Before (Spring Boot 2.x)

```java
@Configuration
@EnableWebSecurity
@EnableGlobalMethodSecurity(prePostEnabled = true)
public class SecurityConfig extends WebSecurityConfigurerAdapter {

    @Autowired
    private UserDetailsService userDetailsService;

    @Bean
    public PasswordEncoder passwordEncoder() {
        return new BCryptPasswordEncoder();
    }

    @Override
    protected void configure(AuthenticationManagerBuilder auth) throws Exception {
        auth.userDetailsService(userDetailsService)
            .passwordEncoder(passwordEncoder());
    }

    @Override
    @Bean
    public AuthenticationManager authenticationManagerBean() throws Exception {
        return super.authenticationManagerBean();
    }

    @Override
    protected void configure(HttpSecurity http) throws Exception {
        http
            .csrf().disable()
            .sessionManagement()
                .sessionCreationPolicy(SessionCreationPolicy.STATELESS)
            .and()
            .exceptionHandling()
                .authenticationEntryPoint((request, response, ex) -> {
                    response.sendError(HttpServletResponse.SC_UNAUTHORIZED, ex.getMessage());
                })
            .and()
            .authorizeRequests()
                .antMatchers(HttpMethod.POST, "/api/users").permitAll()
                .antMatchers("/api/auth/**").permitAll()
                .antMatchers("/h2-console/**").permitAll()
                .antMatchers("/actuator/health").permitAll()
                .anyRequest().authenticated()
            .and()
            .headers().frameOptions().disable();
    }
}
```

### After (Spring Boot 3.x)

```java
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
```

## Servlet Namespace Change

Don't forget the servlet import change:

```java
// Before
import javax.servlet.http.HttpServletResponse;

// After
import jakarta.servlet.http.HttpServletResponse;
```

## Testing Security

Update security test annotations if needed:

```java
@SpringBootTest
@AutoConfigureMockMvc
class SecurityTests {

    @Test
    @WithMockUser(roles = "ADMIN")
    void adminEndpoint_withAdminUser_shouldSucceed() {
        // Test implementation
    }
}
```
