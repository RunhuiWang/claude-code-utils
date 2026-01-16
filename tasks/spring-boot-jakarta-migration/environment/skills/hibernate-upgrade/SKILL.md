# Hibernate 5 to 6 Upgrade Skill

## Overview

Spring Boot 3 uses Hibernate 6, which includes significant changes from Hibernate 5. This skill covers the key migration considerations.

## Key Changes

### 1. Package Namespace

Hibernate 6 uses Jakarta Persistence:

```java
// Before (Hibernate 5)
import javax.persistence.*;
import org.hibernate.annotations.*;

// After (Hibernate 6)
import jakarta.persistence.*;
import org.hibernate.annotations.*;
```

### 2. ID Generation Strategies

Hibernate 6 changed the default ID generation strategy:

```java
// Recommended approach for cross-database compatibility
@Id
@GeneratedValue(strategy = GenerationType.IDENTITY)
private Long id;

// Or using sequences (preferred for some databases)
@Id
@GeneratedValue(strategy = GenerationType.SEQUENCE, generator = "user_seq")
@SequenceGenerator(name = "user_seq", sequenceName = "user_sequence", allocationSize = 1)
private Long id;
```

### 3. Dialect Configuration

Hibernate 6 can auto-detect dialects in most cases:

```properties
# Before (often required in Hibernate 5)
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect

# After (Hibernate 6) - often can be removed
# Hibernate auto-detects based on JDBC URL
# Only specify if you need specific behavior
```

If you must specify:
```properties
# Hibernate 6 dialects (some renamed)
spring.jpa.database-platform=org.hibernate.dialect.H2Dialect
spring.jpa.database-platform=org.hibernate.dialect.PostgreSQLDialect
spring.jpa.database-platform=org.hibernate.dialect.MySQLDialect
```

### 4. Query Changes

#### JPQL Changes

Some JPQL syntax has stricter validation:

```java
// Be explicit about entity aliases
@Query("SELECT u FROM User u WHERE u.active = true")
List<User> findActiveUsers();

// Avoid implicit joins - be explicit
@Query("SELECT u FROM User u JOIN u.roles r WHERE r.name = :roleName")
List<User> findByRoleName(@Param("roleName") String roleName);
```

#### Native Queries

Native query result mapping changed:

```java
// Before (Hibernate 5)
@Query(value = "SELECT * FROM users WHERE active = 1", nativeQuery = true)
List<User> findActiveUsersNative();

// After (Hibernate 6) - works the same, but be careful with projections
@Query(value = "SELECT * FROM users WHERE active = true", nativeQuery = true)
List<User> findActiveUsersNative();
```

### 5. Type Mappings

Some type mappings changed:

```java
// Enum mapping - explicit is better
@Enumerated(EnumType.STRING)
@Column(nullable = false)
private Role role;

// Date/Time - use java.time classes
@Column(name = "created_at")
private LocalDateTime createdAt;  // Preferred over java.util.Date
```

### 6. Fetch Strategies

Default fetch behavior remains similar, but be explicit:

```java
@ManyToOne(fetch = FetchType.LAZY)
@JoinColumn(name = "department_id")
private Department department;

@OneToMany(mappedBy = "user", fetch = FetchType.LAZY)
private List<Order> orders;
```

## Deprecation Removals

### Removed in Hibernate 6

1. **`@Type` annotation** - replaced with `@JdbcTypeCode` or custom type definitions
2. **Legacy ID generators** - use standard JPA generation
3. **`@TypeDef`** - removed, use `@Type` with explicit class

### Example Migration

```java
// Before (Hibernate 5 with custom type)
@TypeDef(name = "json", typeClass = JsonType.class)
@Type(type = "json")
private JsonNode metadata;

// After (Hibernate 6)
@JdbcTypeCode(SqlTypes.JSON)
private JsonNode metadata;
```

## Configuration Properties

Some Hibernate properties changed:

```properties
# Common properties that remain the same
spring.jpa.hibernate.ddl-auto=create-drop
spring.jpa.show-sql=true

# Properties that may need review
spring.jpa.properties.hibernate.format_sql=true
spring.jpa.properties.hibernate.use_sql_comments=true

# New in Hibernate 6
spring.jpa.properties.hibernate.timezone.default_storage=NORMALIZE
```

## Testing Considerations

1. **Entity validation** - Hibernate 6 is stricter about entity mappings
2. **Query parsing** - Some queries that worked in H5 may fail in H6
3. **Performance** - Test performance as query execution plans may differ

## Troubleshooting

### Common Errors

1. **"Unknown entity"** - Ensure entity scanning is configured correctly
2. **"Could not determine type"** - Check type mappings and annotations
3. **"Query syntax error"** - Review JPQL for stricter H6 parsing rules
