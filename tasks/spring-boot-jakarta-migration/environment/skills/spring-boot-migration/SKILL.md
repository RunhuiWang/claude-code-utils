# Spring Boot 2 to 3 Migration Skill

## Overview

This skill provides guidance for migrating Spring Boot applications from version 2.x to 3.x, which is one of the most significant upgrades in Spring Boot history due to the Java EE to Jakarta EE transition.

## Key Migration Steps

### 1. Update Spring Boot Version

Change the parent POM version:

```xml
<!-- Before: Spring Boot 2.7.x -->
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>2.7.18</version>
</parent>

<!-- After: Spring Boot 3.2.x -->
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.0</version>
</parent>
```

### 2. Update Java Version

Spring Boot 3 requires Java 17 or later:

```xml
<properties>
    <!-- Before -->
    <java.version>1.8</java.version>

    <!-- After -->
    <java.version>21</java.version>
</properties>
```

### 3. Remove Deprecated Dependencies

Remove Java EE compatibility dependencies that are no longer needed:

```xml
<!-- Remove these - no longer needed in Spring Boot 3 -->
<dependency>
    <groupId>javax.xml.bind</groupId>
    <artifactId>jaxb-api</artifactId>
</dependency>
```

### 4. Update JWT Library

The old `jjwt` library needs to be replaced with the newer modular version:

```xml
<!-- Before -->
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt</artifactId>
    <version>0.9.1</version>
</dependency>

<!-- After -->
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt-api</artifactId>
    <version>0.12.3</version>
</dependency>
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt-impl</artifactId>
    <version>0.12.3</version>
    <scope>runtime</scope>
</dependency>
<dependency>
    <groupId>io.jsonwebtoken</groupId>
    <artifactId>jjwt-jackson</artifactId>
    <version>0.12.3</version>
    <scope>runtime</scope>
</dependency>
```

## Common Issues

### Issue 1: Compilation Errors After Upgrade

After changing the Spring Boot version, you'll see many compilation errors related to `javax.*` imports. These need to be changed to `jakarta.*` (see Jakarta Namespace skill).

### Issue 2: H2 Database Dialect

The H2 dialect class name changed:

```properties
# Before
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect

# After
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect
# Note: In Hibernate 6, this is often auto-detected and may not need explicit configuration
```

### Issue 3: Actuator Endpoints

Actuator endpoint paths have changed. Review your security configuration if you're exposing actuator endpoints.

## Migration Checklist

- [ ] Update `spring-boot-starter-parent` version to 3.2.x
- [ ] Update `java.version` to 17 or 21
- [ ] Remove deprecated Java EE dependencies
- [ ] Update JWT library if used
- [ ] Run `mvn clean compile` to identify remaining issues
- [ ] Fix all `javax.*` to `jakarta.*` imports
- [ ] Update deprecated API usage
- [ ] Run tests to verify functionality
