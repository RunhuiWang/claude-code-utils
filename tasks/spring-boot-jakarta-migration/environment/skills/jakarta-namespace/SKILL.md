# Jakarta EE Namespace Migration Skill

## Overview

The transition from Java EE to Jakarta EE is the most visible change when migrating to Spring Boot 3. All `javax.*` packages from Java EE have been renamed to `jakarta.*`.

## Package Mappings

### Persistence (JPA)

```java
// Before
import javax.persistence.*;

// After
import jakarta.persistence.*;
```

Affected annotations:
- `@Entity`, `@Table`, `@Column`
- `@Id`, `@GeneratedValue`
- `@ManyToOne`, `@OneToMany`, `@ManyToMany`, `@OneToOne`
- `@JoinColumn`, `@JoinTable`
- `@PrePersist`, `@PreUpdate`, `@PostLoad`
- `@Enumerated`, `@Temporal`
- `EntityManager`, `EntityManagerFactory`
- `EntityNotFoundException`

### Validation

```java
// Before
import javax.validation.constraints.*;
import javax.validation.Valid;

// After
import jakarta.validation.constraints.*;
import jakarta.validation.Valid;
```

Affected annotations:
- `@Valid`
- `@NotNull`, `@NotBlank`, `@NotEmpty`
- `@Size`, `@Min`, `@Max`
- `@Email`, `@Pattern`
- `@Positive`, `@Negative`
- `@Past`, `@Future`

### Servlet

```java
// Before
import javax.servlet.*;
import javax.servlet.http.*;

// After
import jakarta.servlet.*;
import jakarta.servlet.http.*;
```

Affected classes:
- `HttpServletRequest`, `HttpServletResponse`
- `ServletException`
- `Filter`, `FilterChain`
- `HttpSession`
- `Cookie`

### Annotations

```java
// Before
import javax.annotation.*;

// After
import jakarta.annotation.*;
```

Affected annotations:
- `@PostConstruct`, `@PreDestroy`
- `@Resource`
- `@Generated`

### Transaction

```java
// Before
import javax.transaction.*;

// After
import jakarta.transaction.*;
```

Affected classes:
- `@Transactional` (Note: Spring's `@Transactional` doesn't change)
- `TransactionManager`

## Migration Strategies

### Strategy 1: Find and Replace (Simple)

For most cases, a simple find-and-replace works:

```bash
# Using sed (be careful with this approach)
find . -name "*.java" -exec sed -i 's/javax\.persistence/jakarta.persistence/g' {} +
find . -name "*.java" -exec sed -i 's/javax\.validation/jakarta.validation/g' {} +
find . -name "*.java" -exec sed -i 's/javax\.servlet/jakarta.servlet/g' {} +
find . -name "*.java" -exec sed -i 's/javax\.annotation/jakarta.annotation/g' {} +
```

### Strategy 2: IDE Refactoring

Most IDEs support batch import updates. Use your IDE's "Optimize Imports" feature after updating dependencies.

### Strategy 3: OpenRewrite Recipes

OpenRewrite provides automated migration recipes:

```xml
<plugin>
    <groupId>org.openrewrite.maven</groupId>
    <artifactId>rewrite-maven-plugin</artifactId>
    <version>5.23.1</version>
    <configuration>
        <activeRecipes>
            <recipe>org.openrewrite.java.migrate.jakarta.JavaxMigrationToJakarta</recipe>
        </activeRecipes>
    </configuration>
</plugin>
```

## Common Pitfalls

### Pitfall 1: Mixed Namespaces

Don't mix `javax.*` and `jakarta.*` in the same codebase. All must be migrated together.

### Pitfall 2: Third-Party Libraries

Some older libraries still use `javax.*`. Check that all your dependencies have Jakarta-compatible versions.

### Pitfall 3: Test Classes

Don't forget to update test classes - they often have their own imports.

## Verification

After migration, verify no `javax.*` imports remain:

```bash
# Search for remaining javax imports (should return no results)
grep -r "import javax\." --include="*.java" .

# Specifically check for common ones
grep -r "javax\.persistence" --include="*.java" .
grep -r "javax\.validation" --include="*.java" .
grep -r "javax\.servlet" --include="*.java" .
```

## Critical: Ensure jakarta.persistence Is Present

After migration, **your entity classes MUST use jakarta.persistence imports**. This is a hard requirement for Spring Boot 3.

### Required Changes in Entity Classes

Every JPA entity class must have its imports updated:

```java
// BEFORE (Spring Boot 2.x) - WRONG for Spring Boot 3
import javax.persistence.Entity;
import javax.persistence.Table;
import javax.persistence.Id;
import javax.persistence.GeneratedValue;
import javax.persistence.GenerationType;
import javax.persistence.Column;
import javax.persistence.Enumerated;
import javax.persistence.EnumType;
import javax.persistence.PrePersist;
import javax.persistence.PreUpdate;

// AFTER (Spring Boot 3.x) - REQUIRED
import jakarta.persistence.Entity;
import jakarta.persistence.Table;
import jakarta.persistence.Id;
import jakarta.persistence.GeneratedValue;
import jakarta.persistence.GenerationType;
import jakarta.persistence.Column;
import jakarta.persistence.Enumerated;
import jakarta.persistence.EnumType;
import jakarta.persistence.PrePersist;
import jakarta.persistence.PreUpdate;
```

### Quick Migration Command

To migrate all persistence imports at once:

```bash
# Replace javax.persistence with jakarta.persistence in all Java files
find . -name "*.java" -type f -exec sed -i 's/import javax\.persistence/import jakarta.persistence/g' {} +
```

### Verify jakarta.persistence Is Present

After migration, confirm jakarta imports exist:

```bash
# This should return results showing your entity classes
grep -r "import jakarta\.persistence" --include="*.java" .
```

If this returns no results, the migration is incomplete.
